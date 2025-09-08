from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn

from .utils import (
    EasierDict,
    apply_transformation,
    calculate_centroid,
    normalized_distance,
)


class ObjPose_Opt(nn.Module):
    def __init__(
        self,
        rotation_init: torch.Tensor,
        translation_init: torch.Tensor,
        scaling_init: torch.Tensor,
        human_params: EasierDict,
        object_params: EasierDict,
        img: torch.Tensor,
        silhouette_renderer: nn.Module,
        phong_renderer: nn.Module,
        log_dir: str,
        vars: list[str] = ["pose", "scale"],
    ):
        super().__init__()

        self.writer = SummaryWriter(log_dir=log_dir / "tensorboard")
        self.step = 0

        self.img = img
        self.silhouette_renderer = silhouette_renderer
        self.phong_renderer = phong_renderer

        self.rotation = nn.Parameter(
            rotation_init.clone().float(),
            requires_grad=True if "pose" in vars else False,
        )
        self.translation = nn.Parameter(
            translation_init.clone().float(),
            requires_grad=True if "pose" in vars else False,
        )
        if "scale" in vars:
            self.scale = nn.Parameter(scaling_init.float(), requires_grad=True)
        else:
            self.register_buffer("scale", scaling_init.float())

        hcontact = human_params.contact_verts
        ocontact = object_params.contact_verts

        self.register_buffer("human_contact_probs", hcontact)
        self.register_buffer("object_contact_probs", ocontact)

        bbox_coords = torch.nonzero(object_params.mask)
        min_coords = bbox_coords.min(dim=0)[0]
        max_coords = bbox_coords.max(dim=0)[0]
        target_mask_centroid = (min_coords + max_coords) / 2.0

        # * Extract only the depth values where the mask is True
        valid_mask = object_params.mask.bool()

        self.register_buffer("human_vertices", human_params.vertices)
        self.register_buffer("human_faces", human_params.faces)
        self.register_buffer("hum_centroid_offset", human_params.centroid_offset)
        self.register_buffer("human_normals", human_params.normals)

        self.register_buffer("obj_vertices", object_params.vertices)
        self.register_buffer("obj_faces", object_params.faces)
        self.register_buffer("obj_normals", object_params.normals)

        self.register_buffer("target_mask", valid_mask.float())
        self.register_buffer("target_mask_centroid", target_mask_centroid)

    def _log_scalars_hook(self, scalar_dict: dict) -> None:
        for key, val in scalar_dict.items():
            self.writer.add_scalar(f"scalars/{key}", val, self.step)

    def contact_loss(self, obj_verts: torch.Tensor, human_verts: torch.Tensor) -> torch.Tensor:
        # Compute pairwise distances
        dist = torch.cdist(
            obj_verts.unsqueeze(0),
            human_verts.unsqueeze(0),
        ).squeeze(0)  # Shape: (num_obj_verts, num_human_verts)

        # Compute the outer product of the probabilities
        prob_weights = torch.outer(
            self.object_contact_probs, self.human_contact_probs
        )  # Shape: (num_obj_verts, num_human_verts)

        # Compute the weighted mean distance
        weighted_dist = dist * prob_weights
        weighted_mean_dist = weighted_dist.sum() / prob_weights.sum()

        return weighted_mean_dist

    def forward(self, loss_weights: dict) -> tuple[torch.Tensor, str, dict]:
        loss_dict = {}

        obj_vertices = apply_transformation(
            self.obj_vertices, self.rotation, self.translation, self.scale
        )

        off_h_verts = self.human_vertices + self.hum_centroid_offset
        off_o_verts = obj_vertices + self.hum_centroid_offset

        hardP_render, hardP_render_overlay = self.phong_renderer.render(
            self.img,
            o_vertices=off_o_verts,
            h_vertices=off_h_verts,
        )

        sil_img, depth_img = self.silhouette_renderer.render(
            off_o_verts,
            h_vertices=None,
        )
        current_mask = sil_img[0, ..., 3]
        current_depth = depth_img[0, ..., 0]  # * in range [0, 1]

        # Mask Loss
        if "mask_loss" in loss_weights.keys() and self.step >= loss_weights["mask_loss"]["kick_in"]:
            loss_dict["mask_loss"] = self.mask_loss_iou(current_mask)

        # Centroid Loss
        if (
            "centroid_loss" in loss_weights.keys()
            and self.step >= loss_weights["centroid_loss"]["kick_in"]
        ):
            current_mask_centroid = calculate_centroid(current_mask)
            loss_dict["centroid_loss"] = torch.sum(
                (current_mask_centroid - self.target_mask_centroid) ** 2
            )
        if (
            "contact_loss" in loss_weights.keys()
            and self.step >= loss_weights["contact_loss"]["kick_in"]
        ):
            loss_dict["contact_loss"] = self.contact_loss(obj_vertices, self.human_vertices)

        weighted_losses = {
            key: val * loss_weights[key]["w"]
            for key, val in loss_dict.items()
            if key in loss_weights.keys()
            and loss_weights[key]["kick_in"] >= 0
            and self.step >= loss_weights[key]["kick_in"]  # * Kick in loss after certain steps
        }
        total_loss = sum(weighted_losses.values())

        self._log_scalars_hook(weighted_losses)
        self.step += 1

        pbar_str = f"| loss: {total_loss.item():.4f} |"
        for key, val in weighted_losses.items():
            pbar_str += f"{key.replace('_loss', '')}: {val.item():.4f} | "

        output = {
            "hardP_render": hardP_render[0],
            "hardP_render_overlay": hardP_render_overlay[0, ..., :3],
            "object_vertices": obj_vertices.detach(),
            "current_mask": current_mask,
            "current_mask_centroid": current_mask_centroid,
            "current_depth": current_depth,
            "centroid_distance": normalized_distance(
                current_mask_centroid,
                self.target_mask_centroid,
                self.img.shape[:2],
            ),
        }

        return total_loss, pbar_str, output

    def mask_loss_iou(self, current_mask: torch.Tensor) -> torch.Tensor:
        intersection = torch.sum(current_mask * self.target_mask)
        union = torch.sum(current_mask + self.target_mask)
        return 1 - intersection / union
