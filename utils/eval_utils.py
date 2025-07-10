import os
import sys
import torch
import json
import numpy as np
from sklearn.metrics import roc_auc_score

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
utils_path = os.path.join(project_root, 'utils')
sys.path.append(utils_path)

from utils.utils import IGNORE_LABEL

DIST_MATRIX = torch.tensor(np.load('./data/smpl_neutral_geodesic_dist.npy'))

class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

def SIM(map1, map2, eps=1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    intersection = torch.min(map1, map2)
    return intersection.sum()

def intersectionAndUnionGPU(output, target, K):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape, f"output shape: {output.shape}, target shape: {target.shape}"
    output = output.view(-1)
    target = target.view(-1)
    output[target == IGNORE_LABEL] = IGNORE_LABEL
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def get_segmentation_metrics(output_dict):
    
    pred_masks = output_dict["pred_masks"]
    masks_list = output_dict["gt_masks"][0].int()
    output_list = (pred_masks[0] > 0).int()
    assert len(pred_masks) == 1

    intersection, union, acc_iou, num_masks = 0.0, 0.0, 0.0, 0
    for mask_i, output_i in zip(masks_list, output_list):
        intersection_i, union_i, _ = intersectionAndUnionGPU(
            output_i.contiguous().clone(), mask_i.contiguous(), 2,
        )
        intersection += intersection_i
        union += union_i
        acc_iou += intersection_i / (union_i + 1e-5)
        acc_iou[union_i == 0] += 1.0  # no-object target
        num_masks += 1
    intersection, union, acc_iou = intersection.cpu().numpy()/num_masks, union.cpu().numpy()/num_masks, acc_iou.cpu().numpy()/num_masks
    return intersection, union, acc_iou

def get_h_contact_metrics(contact_gt, contact_pred, threshold=0.5):
    
    # Ensure correct dtypes and shapes
    contact_gt = contact_gt.float()  # Convert to float if it's not already
    contact_pred = contact_pred.float()  # Ensure prediction is float
    
    batch_size = contact_gt.shape[0]

    f1_avg, precision_avg, recall_avg = 0, 0, 0

    for b in range(batch_size):
        # Convert predictions to binary using threshold
        pred_binary = (contact_pred[b] >= threshold).float()
        gt_binary = (contact_gt[b] > 0).float()  # Assuming gt is binary or thresholded

        true_positives = (pred_binary * gt_binary).sum()
        predicted_positives = pred_binary.sum()
        actual_positives = gt_binary.sum()

        precision = true_positives / (predicted_positives + 1e-10)
        recall = true_positives / (actual_positives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        f1_avg += f1.item()
        precision_avg += precision.item()
        recall_avg += recall.item()

    f1_avg /= batch_size
    precision_avg /= batch_size
    recall_avg /= batch_size

    return f1_avg, precision_avg, recall_avg

def get_o_contact_metrics(contact_gt, contact_pred, threshold=0.5):

    contact_gt = contact_gt.float()
    contact_pred = contact_pred.float()

    batch_size = contact_gt.shape[0]

    f1_avg, precision_avg, recall_avg = 0, 0, 0

    for b in range(batch_size):
        pred_binary = (contact_pred[b] >= threshold).float()
        gt_binary = (contact_gt[b] > 0).float()

        true_positives = (pred_binary * gt_binary).sum()
        predicted_positives = pred_binary.sum()
        actual_positives = gt_binary.sum()

        precision = true_positives / (predicted_positives + 1e-10)
        recall = true_positives / (actual_positives + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        f1_avg += f1.item()
        precision_avg += precision.item()
        recall_avg += recall.item()

    f1_avg /= batch_size
    precision_avg /= batch_size
    recall_avg /= batch_size
    
    return f1_avg, precision_avg, recall_avg

def get_h_geo_metric(pred, gt):
    
    gt = gt.detach().cpu()
    pred = pred.detach().cpu()

    dist_matrix = DIST_MATRIX

    false_positive_dist = torch.zeros(gt.shape[0])
    false_negative_dist = torch.zeros(gt.shape[0])
    fp_dist_avg, fn_dist_avg = 0, 0
    
    for b in range(gt.shape[0]):
        gt_columns = dist_matrix[:, gt[b, :]==1] if any(gt[b, :]==1) else dist_matrix
        error_matrix = gt_columns[pred[b, :] >= 0.5, :] if any(pred[b, :] >= 0.5) else gt_columns

        false_positive_dist_ = error_matrix.min(dim=1)[0].mean()
        false_negative_dist_ = error_matrix.min(dim=0)[0].mean()

        false_positive_dist[b] = false_positive_dist_
        false_negative_dist[b] = false_negative_dist_
    
    fp_dist_avg = false_positive_dist.mean().item()
    fn_dist_avg = false_negative_dist.mean().item()

    return fp_dist_avg, fn_dist_avg

def get_o_affordance_metrics(contact_gt, contact_pred):

    contact_gt = contact_gt.float()
    contact_pred = contact_pred.float()
    batch_size = contact_gt.shape[0]
    IOU_THRESHOLD = np.linspace(0, 1, 20)
    
    sim_total, mae_total, auc_total, iou_total = 0, 0, 0, 0
    valid_samples = batch_size  # For handling nan cases
    
    for b in range(batch_size):
        # Similarity and MAE
        sim = SIM(contact_gt[b], contact_pred[b], eps=1e-12)
        mae = torch.sum(torch.abs(contact_gt[b] - contact_pred[b])) / 2048
        
        # Convert ground truth to binary
        contact_gt_b = (contact_gt[b] >= 0.5).int()
        
        # Handle cases where all values are same (all 0s or all 1s)
        unique_values = torch.unique(contact_gt_b)
        if len(unique_values) == 1:  # Only one class present
            print(f"Warning!!!: All values are same in GT - {unique_values.item()}")
            auc_score = float('nan')
            aiou = float('nan')
            valid_samples -= 1
        else:
            try:
                # AUC calculation
                auc_score = roc_auc_score(contact_gt_b.cpu().numpy(), 
                                        contact_pred[b].cpu().numpy())
                
                # IOU calculation
                temp_iou = []
                for thres in IOU_THRESHOLD:
                    pred_binary = (contact_pred[b] >= thres).int()
                    intersect = torch.sum(pred_binary & contact_gt_b)
                    union = torch.sum(pred_binary | contact_gt_b)
                    temp_iou.append(1.*intersect/union)
                temp_iou = torch.tensor(temp_iou)
                aiou = temp_iou.mean()
            except ValueError as e:
                print(f"Warning!!!: AUC / IOU calculation failed with error: {e}")
                auc_score = float('nan')
                aiou = float('nan')
                valid_samples -= 1
        
        # Accumulate metrics
        sim_total += sim.item()
        mae_total += mae.item()
        if not np.isnan(auc_score):
            auc_total += auc_score
        if not np.isnan(aiou):
            iou_total += aiou.item()
    
    # Calculate averages, handling potential zero valid_samples
    sim_avg = sim_total / batch_size
    mae_avg = mae_total / batch_size
    auc_avg = auc_total / max(1, valid_samples)
    iou_avg = iou_total / max(1, valid_samples)
    
    return sim_avg, mae_avg, auc_avg, iou_avg, valid_samples

def get_args_for_eval(args):
    # log_folder = f'runs/{args.version.split("/")[-1]}'
    eval_args = args
    with open(f'{args.version}/pretrained_config.json', 'r') as file:
        pretrained_args = json.load(file, object_hook=lambda d: Config(d))
    args = pretrained_args
    print(f'args: {args}')
    args.local_rank = eval_args.local_rank
    args.version = eval_args.version
    args.log_wandb = eval_args.log_wandb
    args.val_dataset = eval_args.val_dataset
    args.val_batch_size = eval_args.val_batch_size
    args.exp_name = f'eval_{args.exp_name}'
    args.disp_size = max(512, eval_args.disp_size)
    args.dataset_dir = './data'
    args.dataset = 'vqa' # some random dataset
    args.eval_only = True
    args.train_from_LISA = False
    args.train_from_LLAVA = False
    return args