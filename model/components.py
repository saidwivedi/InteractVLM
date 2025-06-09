# Description: Contains the implementation of the components used in the model

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib as jl

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
data_path = os.path.join(project_root, 'preprocess_data')
sys.path.append(data_path)

from preprocess_data.constants import HUMAN_VIEW_DICT, OBJS_VIEW_DICT
from utils.utils import debug_tensor

def get_initial_weights(model):
    initial_weights = {name: param.detach().cpu() for name, param in model.named_parameters()}
    return initial_weights

def check_weight_changes(current_model, initial_weights, tag):
    if initial_weights is None:
        return False
    changes, threshold, no_change = {}, 1e-6, True
    for name, param in current_model.named_parameters():
        current_weight = param.detach().cpu()
        initial_weight = initial_weights[name].cpu()
        changes[name] = torch.norm(current_weight - initial_weight).item()  # Calculate the difference norm
    for name, change in changes.items():
        if change > threshold:
            no_change = False
            print(f"{tag} --> Weight change in {name}: {change:.6f}")
    if no_change: 
        print(f"No weight changes detected in {tag}")


class UncertaintyModule(nn.Module):
    def __init__(self, in_channels=256, height=64, width=64):
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.scale_factor = 3
        
        # TODO: Check why Conv2d is giving NaN values
        self.linear1 = nn.Linear(in_channels, 64)
        self.linear2 = nn.Linear(64, 16)
        self.linear3 = nn.Linear(16, 1)
        self.softplus = nn.Softplus()

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.bfloat16()

        # Reshape for linear layers
        x = x.permute(0, 2, 3, 1)  # [batch_size, height, width, channels]
        x = x.reshape(batch_size * self.height * self.width, self.in_channels)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.softplus(self.linear3(x))

        # Reshape back to 2D
        x = x.reshape(batch_size, self.height, self.width, 1)
        x = x.permute(0, 3, 1, 2)  # [batch_size, 1, height, width]

        return x
    
class NumericallyStableMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, need_weights=False):
        batch_size = query.shape[0]
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        output = self.out_proj(attn_output)
        
        return output, attn_weights if need_weights else None
    

class LLaVASAMFusion(nn.Module):
    def __init__(self, sam_embed_dim=256, llava_embed_dim=5120, fusion_dim=128):
        super().__init__()
        self.sam_proj = nn.Linear(sam_embed_dim, fusion_dim)
        self.llava_proj = nn.Linear(llava_embed_dim, fusion_dim)
        self.fusion = NumericallyStableMultiheadAttention(fusion_dim, num_heads=8)
        self.output_proj = nn.Linear(fusion_dim, sam_embed_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
    
    
    def forward(self, sam_embeddings, llava_features):
        
        # Ensure inputs are in bfloat16
        sam_embeddings = sam_embeddings.bfloat16()
        llava_features = llava_features.bfloat16()
        
        # Reshape sam_embeddings
        B, C, H, W = sam_embeddings.shape
        sam_embeddings_reshaped = sam_embeddings.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Project
        sam_proj = self.sam_proj(sam_embeddings_reshaped)
        llava_proj = self.llava_proj(llava_features)
        
        # Perform fusion using multi-head attention
        fused_features, _ = self.fusion(sam_proj, llava_proj, llava_proj)
        
        # Project back to original SAM embedding dimension
        output = self.output_proj(fused_features)
        
        # Reshape output back to original SAM embedding shape
        output = output.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Residual connection with scaling
        return sam_embeddings + output
    
class AttentionSplitter(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.query_human = nn.Linear(hidden_dim, hidden_dim)
        self.query_object = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        
        # Project input to higher dimension
        x_proj = self.input_proj(x)
        
        k = self.key(x_proj)
        v = self.value(x_proj)
        q_human = self.query_human(x_proj)
        q_object = self.query_object(x_proj)
        
        human_attn = F.softmax(torch.matmul(q_human, k.transpose(-2, -1)) / (k.size(-1) ** 0.5), dim=-1)
        object_attn = F.softmax(torch.matmul(q_object, k.transpose(-2, -1)) / (k.size(-1) ** 0.5), dim=-1)
        
        human_output = torch.matmul(human_attn, v)
        object_output = torch.matmul(object_attn, v)
        
        # Project outputs back to original dimension
        human_output = self.output_proj(human_output)
        object_output = self.output_proj(object_output)
        
        return human_output, object_output

class HumanContact3DPredictor(nn.Module):
    def __init__(self, hC_sam_view_type, multiview_channels, threshold=0.3):
        super(HumanContact3DPredictor, self).__init__()
        
        self.hC_sam_view_type = hC_sam_view_type
        self.multiview_channels = multiview_channels
        self.threshold = threshold
        
        metadata_root = './data'
        metadata_folder = HUMAN_VIEW_DICT[hC_sam_view_type]['folder']
        pixel_to_vertex_dict = np.load(os.path.join(metadata_root, metadata_folder, HUMAN_VIEW_DICT[hC_sam_view_type]['pixel_to_vertex']))
        bary_coord_dict = np.load(os.path.join(metadata_root, metadata_folder, HUMAN_VIEW_DICT[hC_sam_view_type]['bary_coords']))
        mask_size = HUMAN_VIEW_DICT[hC_sam_view_type]['mask_size']

        self.views = HUMAN_VIEW_DICT[hC_sam_view_type]['names'].flatten()
        self.num_vertices = HUMAN_VIEW_DICT[hC_sam_view_type]['num_vertices']
        
        pixel_to_vertex_list, bary_coord_list = [], []
        for view in self.views:
            pixel_to_vertex_list.append(pixel_to_vertex_dict[view])
            bary_coord_list.append(bary_coord_dict[view])
        
        self.pixel_to_vertex_map = torch.from_numpy(np.array(pixel_to_vertex_list))
        self.bary_coord_map = torch.from_numpy(np.array(bary_coord_list))

        self.grid_positions = [
            (slice(0, mask_size), slice(0, mask_size)),                     # Human-Top-front
            (slice(0, mask_size), slice(mask_size, 2*mask_size)),           # Human-Bottom-front
            (slice(mask_size, 2*mask_size), slice(0, mask_size)),           # Human-Top-back
            (slice(mask_size, 2*mask_size), slice(mask_size, 2*mask_size))  # Human-Bottom-back
        ]

    def forward(self, seg_maps, ds_names=None):
        device = seg_maps[0].device
        dtype = seg_maps[0].dtype
        batch_size = len(seg_maps)
        ds_names = ds_names if ds_names is not None else ['hcontact'] * batch_size
        
        pred_3d_contacts = torch.zeros((batch_size, self.num_vertices), device=device, dtype=dtype)
        view_count = torch.zeros((batch_size, self.num_vertices), device=device, dtype=dtype)
        
        for b, (seg_map, ds_name) in enumerate(zip(seg_maps, ds_names)):
            if 'hcontact' not in ds_name:
                continue

            if self.multiview_channels == 1: # Grid format (8x1xHxW)
                for view_idx, (row_slice, col_slice) in enumerate(self.grid_positions):
                    view_seg_map = seg_map[0, row_slice, col_slice]
                    self._process_view(view_seg_map, view_idx, b, pred_3d_contacts, view_count)
            else:  # Multiview Channels (8xVxHxW)
                for view_idx in range(self.multiview_channels):
                    view_seg_map = seg_map[view_idx]
                    self._process_view(view_seg_map, view_idx, b, pred_3d_contacts, view_count)
        
        valid_vertices = view_count > 0
        pred_3d_contacts[valid_vertices] /= view_count[valid_vertices]
        
        pred_3d_contacts = (pred_3d_contacts > self.threshold).to(dtype=dtype)
        
        return pred_3d_contacts

    def _process_view(self, view_seg_map, view_idx, batch_idx, pred_3d_contacts, view_count):
        device = view_seg_map.device
        dtype = view_seg_map.dtype
        
        mask = (view_seg_map > self.threshold).to(dtype=dtype)
        
        p2v_map = self.pixel_to_vertex_map[view_idx].to(device)
        bary_coord = self.bary_coord_map[view_idx].to(device, dtype=dtype)
        
        mask_binary = mask > 0
        vertices = p2v_map[mask_binary]
        weights = bary_coord[mask_binary]
        
        vertices = vertices.reshape(-1, 3)
        weights = weights.reshape(-1, 3)
        
        valid_mask = (vertices >= 0) & (vertices < self.num_vertices)
        valid_mask = valid_mask.all(dim=1)
        vertices = vertices[valid_mask]
        weights = weights[valid_mask]
        
        if vertices.numel() == 0:
            return
        
        for i in range(3):
            pred_3d_contacts[batch_idx].scatter_add_(0, vertices[:, i].long(), weights[:, i])
            view_count[batch_idx].scatter_add_(0, vertices[:, i].long(), torch.ones_like(weights[:, i]))

class ObjectPCAfford3DPredictor(nn.Module):
    def __init__(self, oC_sam_view_type, multiview_channels, num_points=2048, threshold=0.3):
        super(ObjectPCAfford3DPredictor, self).__init__()
        
        self.num_points = num_points
        self.multiview_channels = multiview_channels
        self.threshold = threshold
        
        mask_size = OBJS_VIEW_DICT[oC_sam_view_type]['mask_size']

        # For grid format if needed
        self.grid_positions = [
            (slice(0, mask_size), slice(0, mask_size)),                     # Object-Top-front
            (slice(0, mask_size), slice(mask_size, 2*mask_size)),           # Object-Bottom-front
            (slice(mask_size, 2*mask_size), slice(0, mask_size)),           # Object-Top-back
            (slice(mask_size, 2*mask_size), slice(mask_size, 2*mask_size))  # Object-Bottom-back
        ]

    def forward(self, seg_maps, ds_names=None, mask_paths_list=None):
        
        device = seg_maps[0].device
        dtype = seg_maps[0].dtype
        batch_size = len(seg_maps)
        ds_names = ds_names if ds_names is not None else ['oafford'] * batch_size
        
        pred_3d_affordance = torch.zeros((batch_size, self.num_points), device=device, dtype=dtype)
        view_count = torch.zeros((batch_size, self.num_points), device=device, dtype=dtype)
        
        for b, (seg_map, ds_name) in enumerate(zip(seg_maps, ds_names)):
            if 'oafford' not in ds_name:
                continue
            mask_paths = mask_paths_list[b]

            if self.multiview_channels == 1:  # Grid format
                for view_idx, (row_slice, col_slice) in enumerate(self.grid_positions):
                    view_seg_map = seg_map[0, row_slice, col_slice]
                    self._process_view(view_seg_map, view_idx, b, pred_3d_affordance, view_count)
            else:  # Multiview Channels
                for view_idx in range(self.multiview_channels):
                    view_seg_map = seg_map[view_idx]
                    pixel_to_point_map = np.load(mask_paths[view_idx].replace('mask', 'p2pmap')[:-4] + '.npz')
                    self._process_view(view_seg_map, pixel_to_point_map, b, pred_3d_affordance, view_count)
        
        # Normalize by view count
        valid_points = view_count > 0
        pred_3d_affordance[valid_points] /= view_count[valid_points]
        
        return pred_3d_affordance

    def _process_view(self, view_seg_map, pixel_to_point_map, batch_idx, pred_3d_affordance, view_count):
        """
        Process a single view for affordance prediction
        """
        device = view_seg_map.device
        dtype = view_seg_map.dtype
        
        # Get pixel-to-point mapping for this view
        pixel_to_point_map = torch.from_numpy(pixel_to_point_map['mapping']).to(device)
        
        # Get valid pixels
        valid_pixels = pixel_to_point_map != -1
        points = pixel_to_point_map[valid_pixels]
        values = view_seg_map[valid_pixels]
        
        # Initialize view-specific tensors
        view_votes = torch.zeros_like(pred_3d_affordance[batch_idx])
        view_counts = torch.zeros_like(view_count[batch_idx])
        
        # Accumulate votes for this view
        view_votes.scatter_add_(0, points, values)
        view_counts.scatter_add_(0, points, torch.ones_like(values))
        
        # Normalize the votes for this view
        valid_view_points = view_counts > 0
        view_votes[valid_view_points] /= view_counts[valid_view_points]
        
        # Add to total predictions
        pred_3d_affordance[batch_idx] += view_votes
        view_count[batch_idx] += (view_counts > 0).float()


class ObjectMeshContact3DPredictor(nn.Module):
    def __init__(self, oC_sam_view_type, multiview_channels, threshold=0.3):
        super(ObjectMeshContact3DPredictor, self).__init__()
        
        self.multiview_channels = multiview_channels
        self.view_names = OBJS_VIEW_DICT[oC_sam_view_type]['names'].flatten()
        self.threshold = threshold
        
        mask_size = OBJS_VIEW_DICT[oC_sam_view_type]['mask_size']

        self.grid_positions = [
            (slice(0, mask_size), slice(0, mask_size)),                     # Object-Top-front
            (slice(0, mask_size), slice(mask_size, 2*mask_size)),           # Object-Bottom-front
            (slice(mask_size, 2*mask_size), slice(0, mask_size)),           # Object-Top-back
            (slice(mask_size, 2*mask_size), slice(mask_size, 2*mask_size))  # Object-Bottom-back
        ]

    def forward(self, seg_maps, ds_names=None, mask_paths_list=None, lift2d_dict_path=None):
        device = seg_maps[0].device
        dtype = seg_maps[0].dtype

        if 'ocontact' not in ds_names[0]:
            return torch.zeros((1, 0), device=device, dtype=dtype)

        # batch_size should be one since different objects have different number of vertices
        batch_size = len(seg_maps)
        assert batch_size == 1, "Batch size should be 1 since different objects have different number of vertices"

        # TODO: Add provision for inference lift2d_dict_path

        num_vertices = np.load(mask_paths_list[0][0].replace('mask', 'p2vmap').replace('.png', '.npz'))['num_vertices']
        pred_3d_contacts = torch.zeros((batch_size, num_vertices), device=device, dtype=dtype)
        view_count = torch.zeros((batch_size, num_vertices), device=device, dtype=dtype)

        for b, (seg_map, ds_name) in enumerate(zip(seg_maps, ds_names)):
            if 'ocontact' not in ds_name:
                continue
            mask_paths = mask_paths_list[b]

            for view_idx in range(self.multiview_channels):
                view_seg_map = seg_map[view_idx]
                mapping = np.load(mask_paths[view_idx].replace('mask', 'p2vmap').replace('.png', f'.npz'))
                pixel_to_vertex_map = mapping['pixel_to_vertices_map']
                bary_coord_map = mapping['bary_coords_map']
                num_vertices = mapping['num_vertices']
                self._process_view(
                    view_seg_map,
                    pixel_to_vertex_map,
                    bary_coord_map,
                    b,
                    pred_3d_contacts,
                    view_count
                )
        
        valid_vertices = view_count > 0
        pred_3d_contacts[valid_vertices] /= view_count[valid_vertices]
        
        return pred_3d_contacts

    def _process_view(self, view_seg_map, p2v_map, bary_coord, batch_idx, pred_3d_contacts, view_count):
            device = view_seg_map.device
            dtype = view_seg_map.dtype
            
            # Get valid pixels (where mask is non-zero)
            mask = view_seg_map
            # TODO: Check if using sigmoid is correct here
            pred_probs_2d = torch.sigmoid(mask)
            mask_binary = pred_probs_2d > self.threshold
            
            mask_np = mask_binary.cpu().numpy()
            vertices = torch.tensor(p2v_map[mask_np], device=device, dtype=torch.long)
            weights = torch.tensor(bary_coord[mask_np], device=device, dtype=dtype)

            mask_values = pred_probs_2d[mask_binary].reshape(-1, 1)
            
            vertices = vertices.reshape(-1, 3)
            weights = weights.reshape(-1, 3)
            
            # Filter valid vertices
            valid_mask = (vertices >= 0) & (vertices < pred_3d_contacts.shape[1])
            valid_mask = valid_mask.all(dim=1)
            vertices = vertices[valid_mask]
            weights = weights[valid_mask]
            mask_values = mask_values[valid_mask]
            
            if vertices.numel() == 0:
                return

            # Initialize view-specific tensors
            view_votes = torch.zeros_like(pred_3d_contacts[batch_idx])
            view_counts = torch.zeros_like(view_count[batch_idx])
            
            # Accumulate votes for this view using barycentric weights
            for i in range(3):
                view_votes.scatter_add_(0, vertices[:, i].long(), weights[:, i] * mask_values.squeeze())
                view_counts.scatter_add_(0, vertices[:, i].long(), weights[:, i])
            
            # Normalize the votes for this view
            valid_view_vertices = view_counts > 0
            view_votes[valid_view_vertices] /= view_counts[valid_view_vertices]
            
            # Add to total predictions
            pred_3d_contacts[batch_idx] += view_votes
            view_count[batch_idx] += (view_counts > 0).float()

class CamPoseEncoder(nn.Module):
    def __init__(self, input_dim=5, output_dim=256):
        super(CamPoseEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        return x

class ViewIndexCamPoseEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, output_dim=256, num_views=4):
        super().__init__()

        # Spatial understanding
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid(),
        )

        # View-specific transformations
        self.view_transforms = nn.ModuleList([
            nn.Linear(output_dim, output_dim) for _ in range(num_views)
        ])

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, cam_params, view_idx):
        base_encoding = self.spatial_encoder(cam_params)
        view_specific = self.view_transforms[view_idx](base_encoding)
        return view_specific

class VIv1CamPoseEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, output_dim=256, num_views=4):
        super().__init__()

        # Spatial understanding
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # View-specific transformations
        self.view_transforms = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(num_views)
        ])
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, cam_params, view_idx):
        base_encoding = self.spatial_encoder(cam_params)
        view_specific = self.view_transforms[view_idx](base_encoding)
        view_specific = self.sigmoid(view_specific)
        return view_specific