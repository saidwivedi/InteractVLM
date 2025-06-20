import os
import torch
import numpy as np
import cv2
import joblib as jl
import trimesh
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRenderer,
    HardPhongShader,
    TexturesVertex,
    PointLights,
)
from smplx import build_layer

# Import from preprocess_data module
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'preprocess_data'))
from render_mesh_utils import project_vertices_and_create_mask, get_rasterizer

# Constants
LIGHT_LOCATIONS = [[0, 0, 3], [0, 0, 3], [0, 0, -3], [0, 0, -3]]
RENDER_IMG_SIZE = (1024, 1024)
YELLOW_VERTEX_COLOR = [1.00, 0.90, 0.30]
GREY_VERTEX_COLOR = [0.85, 0.85, 0.85]
SMPLX_BODY_MODEL_PATH = '../data/body_models/smplx'


def process_smplx_mesh_with_contacts(contact_vertices_smplx, output_path, contact_threshold=0.1, 
                                   body_model_path=None, gender='neutral'):

    if body_model_path is None:
        body_model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'body_models', 'smplx')
    
    # Load SMPLX body model
    body_model = build_layer(
        model_path=f'{body_model_path}/SMPLX_{gender.upper()}.pkl',
        model_type="smplx",
        use_pca=False,
        gender=gender,
        ext='pkl',
        num_betas=10,
    )
    
    # Create mesh in neutral pose
    with torch.no_grad():
        body_output = body_model()
        vertices = body_output.vertices[0].cpu().numpy()
        faces = body_model.faces.astype(np.int32)
    
    # Process contact vertices
    contact_vertices_smplx = contact_vertices_smplx.cpu().numpy() if torch.is_tensor(contact_vertices_smplx) else contact_vertices_smplx
    
    # Initialize vertex colors (default to light grey)
    vertex_colors = np.ones((vertices.shape[0], 3)) * 0.8  # Light grey
    
    # Apply contact coloring
    if contact_vertices_smplx is not None:
        if contact_vertices_smplx.shape[0] == vertices.shape[0]:
            # Contact vertices is a probability/mask for each vertex
            contact_mask = contact_vertices_smplx > contact_threshold
            # Color contact vertices red
            vertex_colors[contact_mask] = [1.0, 0.1, 0.1]  # Bright red
            
            num_contact_vertices = np.sum(contact_mask)
        else:
            print(f'Warning: Contact vertices shape {contact_vertices_smplx.shape} does not match SMPLX vertices {vertices.shape[0]}')
    
    # Create colored mesh
    colored_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=vertex_colors
    )
    
    # Save the colored mesh
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    colored_mesh.export(output_path)
    print(f'Saved SMPLX body mesh with human contact vertices colored red: {output_path}')


def process_object_mesh_with_contacts(obj_path, contact_vertices, output_path, contact_threshold=0.5):

    # Load the mesh
    mesh = trimesh.load(obj_path)
    
    # Get vertices
    vertices = mesh.vertices
    num_vertices = vertices.shape[0]
    
    # Initialize vertex colors (default to white/grey)
    vertex_colors = np.ones((num_vertices, 3)) * 0.8  # Light grey
    
    # Process contact vertices
    if contact_vertices is not None:
        contact_vertices = contact_vertices.cpu().numpy() if torch.is_tensor(contact_vertices) else contact_vertices
        
        # If contact_vertices is a mask/probability for each vertex
        if contact_vertices.shape[0] == num_vertices:
            contact_mask = contact_vertices > contact_threshold
            # Color contact vertices red
            vertex_colors[contact_mask] = [1.0, 0.1, 0.1]  # Red
        
        # If contact_vertices contains indices of contact vertices
        elif len(contact_vertices.shape) == 1 and contact_vertices.max() < num_vertices:
            contact_indices = contact_vertices[contact_vertices > 0].astype(int)
            if len(contact_indices) > 0:
                vertex_colors[contact_indices] = [1.0, 0.1, 0.1]  # Red
        
        else:
            print(f'Warning: Contact vertices shape {contact_vertices.shape} does not match mesh vertices {num_vertices}')
    
    # Create a new mesh with colors
    colored_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=mesh.faces,
        vertex_colors=vertex_colors
    )
    
    # Save the colored mesh
    colored_mesh.export(output_path)
    print(f'Saved object mesh with contact vertices colored red: {output_path}')
    



def normalize_mesh(vertices, scale_factor=1.0):
    
    # Center the mesh
    centroid = torch.mean(vertices, dim=0)
    vertices_centered = vertices - centroid
    
    # Get the bounding box dimensions
    bbox_min = torch.min(vertices_centered, dim=0)[0]
    bbox_max = torch.max(vertices_centered, dim=0)[0]
    bbox_sizes = bbox_max - bbox_min
    
    # Scale by the largest dimension to maintain aspect ratio
    norm_scale = torch.max(bbox_sizes)
    vertices_normalized = vertices_centered / norm_scale * scale_factor
    
    return vertices_normalized


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


def generate_sam_inp_objs(obj_mesh_f):
    
    print(f'Generating sam_inp_objs for {obj_mesh_f}')
    
    base_folder = os.path.dirname(obj_mesh_f)
    sam_inp_objs = os.path.join(base_folder, 'sam_inp_objs')
    
    if os.path.exists(sam_inp_objs):
        print(f'sam_inp_objs already exists at {sam_inp_objs}')
        return
    
    os.makedirs(sam_inp_objs, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        vertices, faces, aux = load_obj(obj_mesh_f)
        faces = faces.verts_idx
        vertices = normalize_mesh(vertices)
        
        # Define camera views
        views_4 = {
            'frontleft': (1.5, 45, 315, 0., 0.0),
            'frontright': (1.5, 45, 45, 0., 0.0),
            'backleft': (1.5, 330, 135, 0., 0.0),
            'backright': (1.5, 330, 225, 0., 0.0)
        }
        
        # Process both grey and colored versions
        for grey_mesh in [True, False]:
            if grey_mesh:
                vertex_colors = np.array(YELLOW_VERTEX_COLOR)[None, :].repeat(vertices.shape[0], axis=0)
                vertex_colors = torch.tensor(vertex_colors).float()
                prefix = 'grey'
            else:
                min_coords, _ = torch.min(vertices, dim=0)
                max_coords, _ = torch.max(vertices, dim=0)
                vertex_colors = (vertices - min_coords) / (max_coords - min_coords)
                prefix = 'color'
            
            vertex_colors = vertex_colors * 0.8 + 0.1
            
            mesh = Meshes(
                verts=[vertices.to(device)], 
                faces=[faces.to(device)],
                textures=TexturesVertex(verts_features=vertex_colors.unsqueeze(0).to(device))
            )
            
            mask_list, render_list, pixel_to_vertices_map_list, bary_coords_list = [], [], [], []
            contact_vertices = []
            
            for idx, (save_str, camera_params) in enumerate(views_4.items()):
                # Project contact vertices and create mask
                mask, pixel_to_vertices_map, bary_coords = \
                    project_vertices_and_create_mask(
                        mesh, camera_params, contact_vertices, RENDER_IMG_SIZE, min_vertices=3
                    )
                
                render = render_mesh(mesh, camera_params, LIGHT_LOCATIONS[idx], RENDER_IMG_SIZE, device)
                
                mask_list.append(mask)
                render_list.append(render)
                pixel_to_vertices_map_list.append(pixel_to_vertices_map)
                bary_coords_list.append(bary_coords)
            
            # Save rendered images
            for render, view_name in zip(render_list, views_4.keys()):
                output_path = os.path.join(sam_inp_objs, f'obj_render_{prefix}_{view_name}.png')
                cv2.imwrite(output_path, cv2.cvtColor(render, cv2.COLOR_RGB2BGR))
        
        # Save lifting dictionary (only need to do this once)
        lifting_dict = {
            'pixel_to_vertices_map': pixel_to_vertices_map_list,
            'bary_coords_map': bary_coords_list,
            'num_vertices': vertices.shape[0]
        }
        jl.dump(lifting_dict, os.path.join(sam_inp_objs, 'lift2d_dict.pkl'))
        
        print(f'Successfully generated sam_inp_objs at {sam_inp_objs}')
        
    except Exception as e:
        print(f'Error generating sam_inp_objs for {obj_mesh_f}: {str(e)}')
        # Clean up partial results
        if os.path.exists(sam_inp_objs):
            import shutil
            shutil.rmtree(sam_inp_objs)
        raise e