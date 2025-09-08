import os
import random
from datetime import datetime
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from matplotlib import pyplot as plt
from dataclasses import dataclass


@dataclass
class Config:
    human_inference_file: str = "osx_human2.npz"
    human_detection_file: str = "human_detection.json"
    object_mesh_file: str = "object_mesh.obj"
    object_detection_file: str = "object_detection.json"


def matrix_to_rot6d(matrix):
    matrix = matrix.view(-1, 3, 3)
    a1 = matrix[:, :, 0]
    a2 = matrix[:, :, 1]
    # a3 = matrix[:, :, 2]
    return torch.stack((a1, a2), dim=-1).view(-1, 6)


def rot6d_to_matrix(rot_6d):
    rot_6d = rot_6d.view(-1, 3, 2)
    a1 = rot_6d[:, :, 0]
    a2 = rot_6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def normalized_distance(point1, point2, img_shape, device="cuda"):
    point1 = point1 / torch.tensor(img_shape, device=device)
    point2 = point2 / torch.tensor(img_shape, device=device)
    return torch.sqrt(torch.sum((point1 - point2) ** 2)).item()


def calculate_centroid(mask):
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.nelement() == 0:
        return torch.tensor([mask.shape[0] / 2, mask.shape[1] / 2], device=mask.device)
    weights = mask[coords[:, 0], coords[:, 1]]
    # Ensure no in-place operations modify 'coords' or 'weights'
    centroid = torch.sum(coords * weights.unsqueeze(1), dim=0) / torch.sum(weights)
    return centroid


def apply_transformation(vertices, rot6d, translation, scaling=1.0):
    rot_matrix = rot6d_to_matrix(rot6d).view(1, 3, 3)
    scaled_vertices = vertices * scaling
    rotated_vertices = torch.matmul(scaled_vertices.unsqueeze(1), rot_matrix).squeeze(1)
    transformed_vertices = rotated_vertices + translation

    return transformed_vertices


def apply_colormap(tensor):
    cmap = plt.get_cmap("viridis")
    colors = cmap(tensor)[..., :3]  # Get RGB values
    return colors


def save_colored_mesh(vertices, faces, colors, filepath, cmap=False):
    if cmap:
        colors = apply_colormap(colors)
    with open(filepath, "w") as f:
        for i, v in enumerate(vertices):
            f.write(f"v {v[0]} {v[1]} {v[2]} {colors[i][0]} {colors[i][1]} {colors[i][2]}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def save_colored_pc(points, colors, filepath):
    if colors is None:
        colors = np.full_like(points, 0.7)

    if isinstance(colors, torch.Tensor):
        colors = colors.cpu().numpy()

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    if colors.max() <= 1:
        colors = (colors * 255).astype(np.uint8)

    assert (
        trimesh.points.PointCloud(vertices=points, colors=colors).export(
            str(filepath), file_type="ply"
        )
        is not None
    ), f"Failed to export point cloud to {filepath}"


def fix_seeds(seed: int, **kwargs) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if kwargs.get("cudnn", False):
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_timestamp():
    return f"{datetime.now().strftime('%y%m%d-%H%M%S')}"


def human_readable_time(seconds):
    """Convert seconds to a more readable format"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


# ! Adapted from EasyDict package; [pip install easydict]
METHOD_KEYS = ["update", "pop", "to", "to_dict", "detach", "cpu", "cuda"]


class EasierDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and k not in METHOD_KEYS:
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = type(value)(self.__class__(x) if isinstance(x, dict) else x for x in value)
        elif isinstance(value, dict) and not isinstance(value, EasierDict):
            value = EasierDict(value)
        super(EasierDict, self).__setattr__(name, value)
        super(EasierDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, *args):
        if hasattr(self, k):
            delattr(self, k)
        return super(EasierDict, self).pop(k, *args)

    def cpu(self) -> "EasierDict":
        return self.to("cpu")

    def cuda(self) -> "EasierDict":
        return self.to("cuda")

    def detach(self) -> "EasierDict":
        for key in self:
            if isinstance(self[key], torch.Tensor):
                self[key] = self[key].detach()
        return self

    def to(self, device: Union[str, torch.device]) -> "EasierDict":
        for key in self:
            if isinstance(self[key], torch.Tensor):
                self[key] = self[key].to(device)
        return self

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.items():
            if isinstance(v, EasierDict):
                d[k] = v.to_dict()
            elif isinstance(v, (list, tuple)):
                d[k] = type(v)(EasierDict(x).to_dict() if isinstance(x, dict) else x for x in v)
            else:
                d[k] = v
        return d
