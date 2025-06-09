import os
import sys
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
utils_path = os.path.join(project_root, 'utils')
sys.path.append(utils_path)

from utils.utils import IGNORE_LABEL
from utils.utils import debug_tensor

def uncertainty_penalty(uncertainty_map, penalty_weight=0.3):
    # Penalize uncertainties greater than 0
    penalty = torch.mean(uncertainty_map ** 2)
    return penalty_weight * penalty

class CombinedLoss(nn.Module):
    def __init__(self, hCpredictor, oApredictor, oCpredictor,
                 bce_loss_weight=2.0, bce_loss_alpha=0.25,
                 dice_loss_weight=0.5, dice_loss_scale=1000,
                 hC_loss_weight=3.0, oC_loss_weight=1.0,
                 use_uncertainty=False):
        super(CombinedLoss, self).__init__()
        self.bce_loss_weight = bce_loss_weight
        self.bce_loss_alpha = bce_loss_alpha
        self.dice_loss_weight = dice_loss_weight
        self.dice_loss_scale = dice_loss_scale
        self.hC_loss_weight = hC_loss_weight
        self.oC_loss_weight = oC_loss_weight
        self.use_uncertainty = use_uncertainty
        if hCpredictor is not None:
            self.hC_loss_fn = HumanContact3DLoss(hCpredictor)
        if oApredictor is not None:
            self.oA_loss_fn = ObjectAfford3DLoss(oApredictor)
        if oCpredictor is not None:
            self.oC_loss_fn = ObjectContact3DLoss(oCpredictor)

    def forward(self, pred_masks, gt_masks, gt_3d_contacts, mask_paths_list, dataset_names, uncertainty_maps=None):
        
        hC_loss = torch.tensor(0.0, device=pred_masks[0].device, dtype=pred_masks[0].dtype)
        oA_loss = torch.tensor(0.0, device=pred_masks[0].device, dtype=pred_masks[0].dtype)
        oC_loss = torch.tensor(0.0, device=pred_masks[0].device, dtype=pred_masks[0].dtype)
        mask_bce_loss = torch.tensor(0.0, device=pred_masks[0].device, dtype=pred_masks[0].dtype)
        mask_dice_loss = torch.tensor(0.0, device=pred_masks[0].device, dtype=pred_masks[0].dtype)
        mask_uncert_loss = torch.tensor(0.0, device=pred_masks[0].device, dtype=pred_masks[0].dtype)
        mask_l2_loss = torch.tensor(0.0, device=pred_masks[0].device, dtype=pred_masks[0].dtype)
        num_heatmaps, num_binary_masks, num_masks = 0, 0, 0

        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            ds_name = dataset_names[batch_idx]

            if self.use_uncertainty and uncertainty_maps is not None:
                NotImplementedError("Uncertainty loss needs revisit.")
            else:
                if 'oafford' in ds_name: # heatmaps
                    mask_l2_loss += self.compute_mse_mask_loss(pred_mask, gt_mask)
                    num_heatmaps += 1

                mask_bce_loss += self.compute_focal_loss(pred_mask, gt_mask, ds_name).mean()
                mask_dice_loss += self.compute_dice_loss(pred_mask, gt_mask, ds_name, mask_paths_list[batch_idx]).mean()
                num_binary_masks += 1

        # Apply weights and average over batch
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_binary_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_binary_masks + 1e-8)
        mask_l2_loss = self.bce_loss_weight * mask_l2_loss / (num_heatmaps + 1e-8)

        # Compute human contact loss
        if self.hC_loss_weight > 0 and self.hC_loss_fn is not None:
            hC_loss = self.hC_loss_weight * self.hC_loss_fn(pred_masks, gt_3d_contacts, dataset_names)

        # Compute object affordance/contact loss
        if self.oC_loss_weight > 0 and self.oC_loss_fn is not None:
            oA_loss = self.oC_loss_weight * self.oA_loss_fn(pred_masks, gt_3d_contacts, mask_paths_list, dataset_names)

        # Compute object contact loss
        if self.oC_loss_weight > 0 and self.oC_loss_fn is not None:
            oC_loss = self.oC_loss_weight * self.oC_loss_fn(pred_masks, gt_3d_contacts, mask_paths_list, dataset_names)

        # Simple weighted sum
        total_loss = mask_bce_loss + mask_dice_loss + mask_l2_loss + hC_loss + oA_loss + oC_loss + mask_uncert_loss

        return total_loss, mask_bce_loss, mask_dice_loss, mask_l2_loss, hC_loss, oA_loss, oC_loss, mask_uncert_loss
    
    ##################################### MSE Mask Loss #########################################
    def compute_mse_mask_loss(self, pred_mask, gt_mask): # only valid for object contact mask

        def mse_loss_fn(input_view, target_view):
            valid_mask = (target_view != IGNORE_LABEL)
            input_view = input_view[valid_mask]
            target_view = target_view[valid_mask]

            if input_view.numel() == 0:
                print(f'\nWarning: No valid pixels in MSE. Returning zero loss.')
                return torch.tensor(0.0, device=input_view.device, dtype=input_view.dtype)
            
            mse_loss = F.mse_loss(input_view, target_view, reduction='none')
            return mse_loss.mean()
        
        # Process each view separately for multi-view data
        losses = []
        for v in range(pred_mask.shape[0]):
            input_view = pred_mask[v].view(-1)
            target_view = gt_mask[v].view(-1)
            view_loss = mse_loss_fn(input_view, target_view)
            losses.append(view_loss)
        return torch.stack(losses).mean()

    ##################################### FOCAL LOSS #############################################
    def compute_focal_loss(self, inputs, targets, ds_name, gamma=2.0):

        def focal_loss_fn(input_view, target_view):
            valid_mask = (target_view != IGNORE_LABEL)
            input_view = input_view[valid_mask]
            target_view = target_view[valid_mask]

            if input_view.numel() == 0:
                if 'hcontact' in ds_name and 'ocontact' in ds_name:
                    print(f'\nWarning: No valid pixels in focal loss. Returning zero loss.')
                return torch.tensor(0.0, device=input_view.device, dtype=input_view.dtype)
            
            # Object contact are already in (0,1) range
            if 'oafford' in ds_name:
                bce_loss = F.binary_cross_entropy(input_view, target_view, reduction='none')
            else:
                bce_loss = F.binary_cross_entropy_with_logits(input_view, target_view, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = self.bce_loss_alpha * (1 - pt) ** gamma * bce_loss
            return focal_loss.mean()

        # Process each view separately for multi-view data
        if 'hcontact' in ds_name or 'ocontact' in ds_name or 'oafford' in ds_name:
            losses = []
            for v in range(inputs.shape[0]):
                input_view = inputs[v].view(-1)
                target_view = targets[v].view(-1)
                view_loss = focal_loss_fn(input_view, target_view)
                losses.append(view_loss)
            
            return torch.stack(losses).mean()
        # For repeated single-view data, calculate loss once and return
        else:
            inputs = inputs.reshape(-1)
            targets = targets.reshape(-1)
            loss = focal_loss_fn(inputs, targets)
            return loss
    
    ##################################### DICE LOSS #############################################
    def compute_dice_loss(self, inputs, targets, ds_name, mask_path, eps=1e-5):
        # Object contact are already in (0,1) range
        if 'oafford' not in ds_name:
            inputs = inputs.sigmoid()
        scale = self.dice_loss_scale
        
        if 'hcontact' in ds_name or 'ocontact' in ds_name or 'oafford' in ds_name:
            losses = []
            for v in range(inputs.shape[0]):
                input_view = inputs[v].view(-1)
                target_view = targets[v].view(-1)
                
                valid_mask = (target_view != IGNORE_LABEL)
                input_view = input_view[valid_mask]
                target_view = target_view[valid_mask]

                if target_view.numel() == 0 or target_view.sum() == 0:
                    return torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)

                numerator = 2 * (input_view / scale * target_view).sum()
                denominator = (input_view / scale).sum() + (target_view / scale).sum()
                view_loss = 1 - (numerator + eps) / (denominator + eps)
                
                if torch.isnan(view_loss):
                    print(f"\nNaN detected in dice loss:")
                    print(f"Input range: [{input_view.min():.6f}, {input_view.max():.6f}]")
                    print(f"Target range: [{target_view.min():.6f}, {target_view.max():.6f}]")
                    print(f"Numerator: {numerator:.6f}, Denominator: {denominator:.6f}")
                    # Return zero loss for this iteration to continue training
                    view_loss = torch.tensor(0.0, device=inputs.device)

                losses.append(view_loss)
            
            total_loss = torch.stack(losses).mean()
            if total_loss.item() == 0:
                print(f"\nWarning: Zero dice loss detected - {mask_path}")
            return torch.stack(losses).mean()
        else:
            inputs = inputs.reshape(-1)
            targets = targets.reshape(-1)
            
            numerator = 2 * (inputs / scale * targets).sum()
            denominator = (inputs / scale).sum() + (targets / scale).sum()
            loss = 1 - (numerator + eps) / (denominator + eps)
            return loss

    def compute_uncertainty_loss(self, uncertainty_map):
        return torch.mean(F.relu(uncertainty_map))


class HumanContact3DLoss(nn.Module):
    def __init__(self, predictor, focal_alpha=0.25, focal_gamma=2.0, sparsity_weight=0.01):
        super(HumanContact3DLoss, self).__init__()
        
        self.predictor = predictor
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.sparsity_weight = sparsity_weight

    def forward(self, seg_maps, gt_3d_contacts, dataset_names):
        dtype = seg_maps[0].dtype
        device = seg_maps[0].device

        hcontact_mask = ['hcontact' in ds for ds in dataset_names]
        if not any(hcontact_mask):
            return torch.tensor(0.0, device=device, dtype=dtype)
        
        seg_maps = [seg_maps[i] for i, mask in enumerate(hcontact_mask) if mask]
        gt_3d_contacts = torch.stack([gt_3d_contacts[i].to(dtype) for i, mask in enumerate(hcontact_mask) if mask])
        
        pred_3d_contacts = self.predictor(seg_maps)

        # Handle case where no vertices were selected in any view
        if pred_3d_contacts.sum() == 0:
            print("Warning: No vertices were selected in any view. Returning zero loss.")
            return torch.tensor(0.0, device=device, dtype=dtype)
        
        bce_loss = F.binary_cross_entropy(pred_3d_contacts, gt_3d_contacts, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.focal_alpha * (1-pt)**self.focal_gamma * bce_loss
        
        sparsity_loss = pred_3d_contacts.mean()
        
        total_loss = focal_loss.mean() + self.sparsity_weight * sparsity_loss
        
        return total_loss

class ObjectContact3DLoss(nn.Module):
    def __init__(self, predictor, focal_alpha=0.25, focal_gamma=2.0, sparsity_weight=0.01):
        super(ObjectContact3DLoss, self).__init__()
        self.predictor = predictor
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.sparsity_weight = sparsity_weight

    def forward(self, seg_maps, gt_3d_contacts, mask_paths_list, dataset_names):
        dtype = seg_maps[0].dtype
        device = seg_maps[0].device

        hcontact_mask = ['ocontact' in ds for ds in dataset_names]
        if not any(hcontact_mask):
            print("Warning: No object contact samples found. Returning zero loss.")
            return torch.tensor(0.0, device=device, dtype=dtype)

        losses = []

        for i, mask in enumerate(hcontact_mask):
            if not mask:
                continue

            seg_map = seg_maps[i]
            gt_contact = gt_3d_contacts[i].to(dtype)
            pred_contact = self.predictor([seg_map], ['ocontact'], [mask_paths_list[i]])[0]

            if pred_contact.sum() == 0:
                print(f"Warning: No vertices were selected in sample {i}. Skipping.")
                continue

            pred_contact_probs = torch.clamp(pred_contact, 1e-6, 1.0 - 1e-6)
            bce_loss = F.binary_cross_entropy(pred_contact_probs, gt_contact, reduction='none')

            pt = torch.exp(-bce_loss)
            focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce_loss
            sparsity_loss = pred_contact_probs.mean()
            total_loss = focal_loss.mean() + self.sparsity_weight * sparsity_loss

            losses.append(total_loss)

        if not losses:
            print("Warning: All predictions are empty. Returning zero loss.")
            return torch.tensor(0.0, device=device, dtype=dtype)

        return torch.stack(losses).mean()


class ObjectAfford3DLoss(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.gamma = 2
        self.alpha = 0.25
    
    def forward(self, seg_maps, gt_3d_contacts, mask_paths_list, dataset_names):
        dtype = seg_maps[0].dtype
        device = seg_maps[0].device
        
        # Filter object contact samples
        oafford_mask = ['oafford' in ds for ds in dataset_names]
        if not any(oafford_mask):
            return torch.tensor(0.0, device=device, dtype=dtype)
        
        # Filter inputs
        seg_maps = [seg_maps[i] for i, mask in enumerate(oafford_mask) if mask]
        mask_paths_list = [mask_paths_list[i] for i, mask in enumerate(oafford_mask) if mask]
        gt_3d_contacts = torch.stack([gt_3d_contacts[i].to(dtype) for i, mask in enumerate(oafford_mask) if mask])
        
        # Seg_maps are already in (0,1) and hence pred_3d_contacts will also be in (0,1)
        pred_3d_contacts = self.predictor(seg_maps, mask_paths_list)

        # Clamp for numerical stability
        pred_3d_contacts = torch.clamp(pred_3d_contacts, 1e-6, 1-1e-6)

        if pred_3d_contacts.sum() == 0:
            print("Warning: No points predicted as affordance")
            return torch.tensor(0.0, device=device, dtype=dtype)

        # Focal Loss part
        temp1 = -(1-self.alpha)*torch.mul(pred_3d_contacts**self.gamma, 
                           torch.mul(1-gt_3d_contacts, torch.log(1-pred_3d_contacts)))
        temp2 = -self.alpha*torch.mul((1-pred_3d_contacts)**self.gamma, 
                           torch.mul(gt_3d_contacts, torch.log(pred_3d_contacts)))
        CELoss = torch.sum(torch.mean(temp1 + temp2, (0, 1)))
        
        # DICE Loss part
        intersection_positive = torch.sum(pred_3d_contacts*gt_3d_contacts, 1)
        cardinality_positive = torch.sum(torch.abs(pred_3d_contacts)+torch.abs(gt_3d_contacts), 1)
        dice_positive = (intersection_positive+1e-6)/(cardinality_positive+1e-6)
        
        intersection_negative = torch.sum((1.-pred_3d_contacts)*(1.-gt_3d_contacts), 1)
        cardinality_negative = torch.sum(2-torch.abs(pred_3d_contacts)-torch.abs(gt_3d_contacts), 1)
        dice_negative = (intersection_negative+1e-6)/(cardinality_negative+1e-6)
        
        DICELoss = torch.sum(torch.mean(1.5-dice_positive-dice_negative, 0))

        mse_loss = F.mse_loss(pred_3d_contacts, gt_3d_contacts) * 0.8
        l1_loss = F.l1_loss(pred_3d_contacts, gt_3d_contacts) * 0.4
        
        # Original classification losses with reduced weights
        CELoss = CELoss * 0.5
        DICELoss = DICELoss * 0.3
    
        
        return CELoss + DICELoss + mse_loss + l1_loss