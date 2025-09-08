import json
import os
from os.path import dirname, join

import cv2
import numpy as np
import torch
import trimesh
from pytorch3d.structures import Meshes

from .constants import (
    OSX_FOCAL_VIRTUAL,
    OSX_INPUT_BODY_SHAPE,
    OSX_PRINCPT,
)


class HumanParams:
    def __init__(
        self,
        vertices,
        faces,
        contact_verts,
        centroid_offset,
        bbox,
        mask,
        human_parts_dict,
    ):
        self.vertices = vertices
        self.faces = faces
        self.contact_verts = contact_verts
        self.centroid_offset = centroid_offset
        self.bbox = bbox
        self.mask = mask
        self.human_parts_dict = human_parts_dict

    def to_cuda(self):
        self.vertices = torch.from_numpy(self.vertices).float().cuda()
        self.faces = torch.from_numpy(self.faces).float().cuda()
        normals = Meshes(verts=[self.vertices], faces=[self.faces]).verts_normals_list()[0]
        normals /= torch.norm(normals, dim=1, keepdim=True)
        self.normals = normals.float().cuda()
        self.contact_verts = torch.from_numpy(self.contact_verts).float().cuda()
        self.centroid_offset = torch.from_numpy(self.centroid_offset).float().cuda()
        self.bbox = torch.from_numpy(self.bbox).float().cuda()
        self.mask = torch.from_numpy(self.mask).cuda()


class ObjectParams:
    def __init__(
        self,
        vertices,
        faces,
        contact_verts,
        bbox,
        mask,
        scale,
    ):
        self.vertices = vertices
        self.faces = faces
        self.contact_verts = contact_verts
        self.bbox = bbox
        self.mask = mask
        self.scale = scale

    def to_cuda(self):
        self.vertices = torch.from_numpy(self.vertices).float().cuda()
        self.faces = torch.from_numpy(self.faces).float().cuda()
        normals = Meshes(verts=[self.vertices], faces=[self.faces]).verts_normals_list()[0]
        normals /= torch.norm(normals, dim=1, keepdim=True)
        self.normals = normals.float().cuda()
        self.contact_verts = torch.from_numpy(self.contact_verts).float().cuda()
        self.bbox = torch.from_numpy(self.bbox).float().cuda()
        self.mask = torch.tensor(self.mask).cuda()
        self.scale = torch.tensor(self.scale).float().cuda()


class CameraParams:
    def __init__(self, focal_length, principal_point):
        self.focal_length = focal_length
        self.principal_point = principal_point

    def to_cuda(self):
        self.focal_length = (
            torch.tensor(self.focal_length).float().cuda()
            if not isinstance(self.focal_length, torch.Tensor)
            else self.focal_length
        )
        self.principal_point = (
            torch.tensor(self.principal_point).float().cuda()
            if not isinstance(self.principal_point, torch.Tensor)
            else self.principal_point
        )


def get_camera_params_torch(human_bbox):
    fv1 = torch.tensor(OSX_FOCAL_VIRTUAL[0], dtype=torch.float32, device="cuda")
    fv2 = torch.tensor(OSX_FOCAL_VIRTUAL[1], dtype=torch.float32, device="cuda")
    bs1 = torch.tensor(OSX_INPUT_BODY_SHAPE[0], dtype=torch.float32, device="cuda")
    bs2 = torch.tensor(OSX_INPUT_BODY_SHAPE[1], dtype=torch.float32, device="cuda")
    pt1 = torch.tensor(OSX_PRINCPT[0], dtype=torch.float32, device="cuda")
    pt2 = torch.tensor(OSX_PRINCPT[1], dtype=torch.float32, device="cuda")

    focal = torch.stack([fv1 / bs2 * human_bbox[2], fv2 / bs1 * human_bbox[3]], dim=0)
    princpt = torch.stack(
        [pt1 / bs2 * human_bbox[2] + human_bbox[0], pt2 / bs1 * human_bbox[3] + human_bbox[1]],
        dim=0,
    )
    return focal, princpt


def load_image(
    image_path: str,
    image_size: int = None,
    im_type_flag: int = cv2.COLOR_BGR2RGB,
) -> np.ndarray:
    if not isinstance(image_path, str):
        image_path = str(image_path)
    image = cv2.imread(image_path)
    assert image is not None, f"Image not found at {image_path}"
    image = cv2.cvtColor(image, im_type_flag)

    # resize image
    if image_size is not None:
        h, w = image.shape[:2]
        r = min(image_size / w, image_size / h)
        w = int(r * w)
        h = int(r * h)
        image = cv2.resize(image, (w, h))

    return image


def load_params(
    human_inference_file: str,
    object_mesh_file: str,
    object_detection_file: str,
) -> tuple:
    root_folder = dirname(human_inference_file)

    # * Load human parameters
    human_npz = np.load(human_inference_file, allow_pickle=True)
    hum_vertices = torch.tensor(
        human_npz["smpl_vertices"][0],
        dtype=torch.float32,
        device="cuda",
    )

    faces = human_npz["smpl_faces"].astype(int)

    hum_contacts = np.load(join(root_folder, "hcontact_vertices.npz"))["pred_contact_3d_smplx"]
    if os.path.isfile(join(root_folder, "human_mask.png")):
        human_mask = load_image(
            join(root_folder, "human_mask.png"),
            im_type_flag=cv2.COLOR_BGR2GRAY,
        )
    else:
        human_mask = np.array(
            json.load(open(join(root_folder, "human_detection.json"), "r"))["mask"],
            dtype=np.uint8,
        )

        cv2.imwrite(join(root_folder, "human_mask.png"), human_mask * 255)

    # * Load camera parameters
    foc_osx, princpt = get_camera_params_torch(human_npz["bbox_2"][0])

    foc = foc_osx.clone()
    camera_params = CameraParams(focal_length=foc, principal_point=princpt)
    camera_params.to_cuda()

    # Rescale the mesh to fit the new focal length
    human_mesh = trimesh.Trimesh(vertices=hum_vertices.cpu(), faces=faces, process=False)
    # save the centroid offset
    centroid_offset = human_mesh.centroid
    # center the mesh
    human_mesh.apply_translation(-centroid_offset)
    human_params = HumanParams(
        vertices=human_mesh.vertices,
        faces=faces,
        contact_verts=hum_contacts,
        centroid_offset=centroid_offset.copy(),
        bbox=human_npz["bbox_2"][0],
        mask=human_mask,
        human_parts_dict=None,
    )
    human_params.to_cuda()

    # * Load object parameters
    obj_mesh = trimesh.load(object_mesh_file, process=False)
    # center the mesh
    obj_mesh.apply_translation(-obj_mesh.centroid)
    obj_mesh.vertices[:, 1] *= -1
    obj_mesh.vertices[:, 2] *= -1

    detection = json.load(open(object_detection_file, "r"))

    root_folder = dirname(object_detection_file)
    object_contact_verts = np.load(join(root_folder, "ocontact_vertices.npz"))["pred_contact_3d"]
    if len(object_contact_verts.shape) == 2:
        object_contact_verts = object_contact_verts.squeeze(0)

    if os.path.exists(join(root_folder, "object_mask.png")):
        obj_mask = load_image(join(root_folder, "object_mask.png"), im_type_flag=cv2.COLOR_BGR2GRAY)
    else:
        obj_mask = np.array(detection["mask"], dtype=np.uint8).squeeze()

    object_params = ObjectParams(
        vertices=obj_mesh.vertices,
        faces=obj_mesh.faces,
        contact_verts=object_contact_verts,
        bbox=np.array(detection["bbox"]),
        mask=obj_mask,
        scale=np.array([1.0]),
    )

    object_params.to_cuda()
    return human_params, object_params, camera_params
