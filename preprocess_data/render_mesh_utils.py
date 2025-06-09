import os
import numpy as np
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import json
import cv2
import joblib
from smplx import build_layer
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    TexturesVertex,
    PointLights,
)
from pytorch3d.transforms import RotateAxisAngle

FACES_PER_PIXEL = 1
BLUR_RADIUS = 0.0

def euler_to_quaternion(r):
    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]

    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = torch.cos(z)
    sz = torch.sin(z)
    cy = torch.cos(y)
    sy = torch.sin(y)
    cx = torch.cos(x)
    sx = torch.sin(x)
    quaternion = torch.zeros_like(r.repeat(1,2))[..., :4].to(r.device)
    quaternion[..., 0] += cx*cy*cz - sx*sy*sz
    quaternion[..., 1] += cx*sy*sz + cy*cz*sx
    quaternion[..., 2] += cx*cz*sy - sx*cy*sz
    quaternion[..., 3] += cx*cy*sz + sx*cz*sy
    return quaternion

def quaternion_to_rotation_matrix(quat):
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def batch_euler2matrix(r):
    return quaternion_to_rotation_matrix(euler_to_quaternion(r))

def get_virtuvian_body_pose():
    pose = torch.zeros([21, 3], dtype=torch.float32)
    angle = 30 * np.pi / 180.
    pose[0, 2] = angle
    pose[1, 2] = -angle
    return batch_euler2matrix(pose).unsqueeze(0)

def compute_vertex_normals(verts, faces):
    """Compute vertex normals for the mesh."""
    # Initialize a tensor to hold the normals
    normals = torch.zeros_like(verts)

    # Compute face normals
    v0 = verts[faces[:, 1]] - verts[faces[:, 0]]
    v1 = verts[faces[:, 2]] - verts[faces[:, 0]]
    face_normals = torch.cross(v0, v1, dim=1)

    # Accumulate face normals to vertices
    normals.index_add_(0, faces[:, 0], face_normals)
    normals.index_add_(0, faces[:, 1], face_normals)
    normals.index_add_(0, faces[:, 2], face_normals)

    # Normalize the normals
    normals = torch.nn.functional.normalize(normals, p=2, dim=1)

    return normals

def get_body_params(body_params):
    body_params_tensor = {}
    for k, v in body_params.items():
        v = torch.tensor(np.array(v), dtype=torch.float32)
        if k == 'shape':
            v = v[:10]
        if 'pose' in k:
            v = v.view(-1, 3, 3)
        if 'global_orient' in k:
            v = v.view(3, 3)
        v = v.unsqueeze(0)
        body_params_tensor[k] = v
    return body_params_tensor

def apply_rotation_to_mesh(mesh, angle_degrees, axis, device='cuda'):
    R = RotateAxisAngle(angle=angle_degrees, axis=axis, device=device)
    verts = mesh.verts_packed()
    rotated_verts = R.transform_points(verts)
    return Meshes(verts=[rotated_verts], faces=[mesh.faces_packed()], textures=mesh.textures)

def get_rasterizer(camera_params, image_size=(512, 512), device='cuda'):
    distance, elevation, azimuth, x_trans, y_trans = camera_params
    R, T = look_at_view_transform(distance, elevation, azimuth)
    T[0, 1] += y_trans
    T[0, 0] += x_trans
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=BLUR_RADIUS,
        faces_per_pixel=FACES_PER_PIXEL,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    return rasterizer, cameras

def project_vertices_and_create_mask(mesh, camera_params, contact_vertices, image_size=(512, 512), min_vertices=3, device='cuda'):
    rasterizer, _ = get_rasterizer(camera_params, image_size, device)
    fragments = rasterizer(mesh)
    pixel_to_face = fragments.pix_to_face[0].cpu().numpy()  # Shape: (H, W, faces_per_pixel)
    bary_coords = fragments.bary_coords[0].cpu().numpy()  # Shape: (H, W, faces_per_pixel, 3)
    
    faces = mesh.faces_packed().cpu().numpy()
    contact_vertices_set = set(contact_vertices)
    
    # Create a mapping from face index to a boolean indicating if at least two of its vertices are contact vertices
    face_contact_count = np.sum(np.isin(faces, list(contact_vertices_set)), axis=1)
    face_more_than_min_contact = face_contact_count >= min_vertices
    
    # Create the mask using numpy operations
    mask = np.any((face_more_than_min_contact[pixel_to_face] & (pixel_to_face >= 0)), axis=-1) * 255
    
    # Create pixel_to_vertices_map
    pixel_to_vertices_map = np.full(image_size + (3,), -1, dtype=np.int64)
    valid_pixels = np.any(pixel_to_face >= 0, axis=-1)
    
    # Flatten the valid_pixels and pixel_to_face arrays
    valid_pixels_flat = valid_pixels.flatten()
    pixel_to_face_flat = pixel_to_face.reshape(-1, FACES_PER_PIXEL)
    
    # Get the face indices for valid pixels (first valid face per pixel)
    valid_face_indices = np.where(pixel_to_face_flat >= 0, pixel_to_face_flat, -1).max(axis=1)
    valid_face_indices = valid_face_indices[valid_pixels_flat]
    
    # Get the corresponding vertices for these faces
    valid_vertices = faces[valid_face_indices]
    
    # Create a flattened version of pixel_to_vertices_map
    pixel_to_vertices_map_flat = pixel_to_vertices_map.reshape(-1, 3)
    
    # Assign the valid vertices to the flattened pixel_to_vertices_map
    pixel_to_vertices_map_flat[valid_pixels_flat] = valid_vertices
    
    # Reshape pixel_to_vertices_map back to its original shape
    pixel_to_vertices_map = pixel_to_vertices_map_flat.reshape(image_size + (3,))
    
    # For bary_coords, we'll use the coordinates of the first valid face per pixel
    bary_coords_flat = bary_coords.reshape(-1, FACES_PER_PIXEL, 3)
    valid_bary_coords = bary_coords_flat[np.arange(len(bary_coords_flat)), np.argmax(pixel_to_face_flat >= 0, axis=1)]
    valid_bary_coords = valid_bary_coords.reshape(image_size + (3,))
    
    return mask.astype(np.uint8), pixel_to_vertices_map, valid_bary_coords


def render_mesh(mesh, camera_params, light_location, image_size=(512, 512), device='cuda'):
    rasterizer, cameras = get_rasterizer(camera_params, image_size, device)
    fill_light = PointLights(
        device=device, 
        ambient_color=[[0.5, 0.5, 0.5]],
        diffuse_color=[[0.3, 0.3, 0.3]],
        specular_color=[[0.2, 0.2, 0.2]],
        location=[light_location]
    )
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=fill_light,
        )
    )

    images = renderer(mesh)
    image = images[0, ..., :3].cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return image

def verify_contact_reconstruction_diff(mask_list, pixel_to_vertices_map_list, bary_coords_list, original_contact_vertices, threshold=0.5, num_vertices=6890, debug=False):
    reconstructed_contact_vertices = np.zeros(num_vertices)
    view_count = np.zeros(num_vertices)
    
    for mask, pixel_to_vertices_map, bary_coords in zip(mask_list, pixel_to_vertices_map_list, bary_coords_list):
        mask_binary = mask > threshold
        vertices = pixel_to_vertices_map[mask_binary]
        weights = bary_coords[mask_binary]
        
        for v, w in zip(vertices, weights):
            reconstructed_contact_vertices[v] += w.sum()
            view_count[v] += 1
    
    # Normalize by the number of views each vertex appeared in
    valid_vertices = view_count > 0
    reconstructed_contact_vertices[valid_vertices] /= view_count[valid_vertices]
    
    # Apply threshold to get final contact vertices
    reconstructed_contact_vertices = (reconstructed_contact_vertices > threshold)
    reconstructed_contact_vertices_set = set(np.where(reconstructed_contact_vertices)[0])
    
    original_contact_vertices_set = set(original_contact_vertices)
    
    correctly_reconstructed = reconstructed_contact_vertices_set.intersection(original_contact_vertices_set)
    missed_vertices = original_contact_vertices_set - reconstructed_contact_vertices_set
    extra_vertices = reconstructed_contact_vertices_set - original_contact_vertices_set

    if debug:        
        print(f'\nThreshold: {threshold}')
        print(f"Original contact vertices: {len(original_contact_vertices_set)}")
        print(f"Reconstructed contact vertices: {len(reconstructed_contact_vertices_set)}")
        print(f"Correctly reconstructed: {len(correctly_reconstructed)}")
        print(f"Missed vertices: {len(missed_vertices)}")
        print(f"Extra vertices: {len(extra_vertices)}")
    
    return reconstructed_contact_vertices_set, missed_vertices, extra_vertices



VIRTUVIAN_POSE = get_virtuvian_body_pose()