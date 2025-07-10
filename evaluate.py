# evaluate.py
import argparse
import os
import sys
import joblib as jl
from functools import partial
import shutil

import deepspeed
import numpy as np
import torch
import tqdm
import json
import transformers
from transformers import BitsAndBytesConfig # For 4/8bit quantization

from torch.utils.tensorboard import SummaryWriter
import wandb
from distutils.util import strtobool


from model.InteractVLM import InteractVLMForCausalLM
from model.llava import conversation as conversation_lib
from datasets.dataset import ValDataset, collate_fn
from datasets.hcontact_3d import HContactSegDataset
from datasets.ocontact_3d import OContactSegDataset, OAffordSegDataset
from utils.eval_utils import (
    get_h_contact_metrics, get_h_geo_metric,
    get_o_contact_metrics, get_o_affordance_metrics,
    get_segmentation_metrics, get_args_for_eval,
)
from utils.utils import (
    get_dtype, dict_to_cuda, add_new_tokens, log_metric, log_images, copy_code,
    AverageMeter, Summary,
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
)
from preprocess_data.constants import HUMAN_VIEW_DICT, OBJS_VIEW_DICT, DAMON_CATEGORIES_MAPPING



def validate(val_loader, model_engine, epoch, loggers, args, ds_name):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    saved_results_hC = {'imgnames': [], 'pred': [], 'gt': [], 'f1': [], 'geo': [], 'objnames': []}
    saved_results_oA = {'imgnames': [], 'pred': [], 'gt': [], 'sim': [], 'mae': [], 'auc': [], 'iou': []}
    saved_results_oC = {'imgnames': [], 'pred': [], 'gt': [], 'f1': []}

    hcontact_metric = False
    if 'hcontact' in ds_name and args.hC_loss_weight > 0:
        hcontact_metric = True
        hcontact_f1_meter = AverageMeter("F1", ":6.3f", Summary.SUM)
        hcontact_precision_meter = AverageMeter("Precision", ":6.3f", Summary.SUM)
        hcontact_recall_meter = AverageMeter("Recall", ":6.3f", Summary.SUM)
        hcontact_geo_meter = AverageMeter("Geo", ":6.3f", Summary.SUM)

    oafford_metric = False
    if 'oafford' in ds_name and args.oC_loss_weight > 0:
        oafford_metric = True
        oafford_sim_meter = AverageMeter("Sim", ":6.3f", Summary.SUM)
        oafford_mae_meter = AverageMeter("MAE", ":6.3f", Summary.SUM)
        oafford_auc_meter = AverageMeter("AUC", ":6.3f", Summary.SUM)
        oafford_iou_meter = AverageMeter("IoU", ":6.3f", Summary.SUM)

    ocontact_metric = False
    if 'ocontact' in ds_name and args.oC_loss_weight > 0:
        ocontact_metric = True
        ocontact_f1_meter = AverageMeter("F1", ":6.3f", Summary.SUM)
        ocontact_precision_meter = AverageMeter("Precision", ":6.3f", Summary.SUM)
        ocontact_recall_meter = AverageMeter("Recall", ":6.3f", Summary.SUM)

    model_engine.eval()

    for batch_idx, input_dict in tqdm.tqdm(enumerate(val_loader)):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        input_dict["images"] = input_dict["images"].to(dtype=get_dtype(args.precision))
        input_dict["images_clip"] = input_dict["images_clip"].to(dtype=get_dtype(args.precision))
        input_dict["gt_contact_3d"] = torch.vstack(input_dict["gt_contact_3d_list"])
        input_dict["cam_params"] = input_dict["cam_params"].to(dtype=get_dtype(args.precision))

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        # segmentation metrics
        intersection, union, acc_iou = get_segmentation_metrics(output_dict)

        # human contact metrics
        if hcontact_metric:
            f1, precision, recall = get_h_contact_metrics(input_dict['gt_contact_3d'], output_dict['pred_human_3d_contact'])
            fp_geo, _ = get_h_geo_metric(input_dict['gt_contact_3d'], output_dict['pred_human_3d_contact'])
            hcontact_f1_meter.update(f1)
            hcontact_precision_meter.update(precision)
            hcontact_recall_meter.update(recall)
            hcontact_geo_meter.update(fp_geo)

            # save results
            saved_results_hC['imgnames'].append(input_dict['image_paths'])
            saved_results_hC['objnames'].append(input_dict['sampled_classes_list'])
            saved_results_hC['pred'].append(output_dict['pred_human_3d_contact'].cpu().numpy())
            saved_results_hC['gt'].append(input_dict['gt_contact_3d'].cpu().numpy())
            saved_results_hC['f1'].append(f1)
            saved_results_hC['geo'].append(fp_geo)

        # object affordance metrics
        if oafford_metric:
            sim, mae, auc, iou, valid_samples = \
                get_o_affordance_metrics(input_dict['gt_contact_3d'], output_dict['pred_object_3d_afford'])
            saved_results_oA['imgnames'].append(os.path.basename(input_dict['image_paths'][0]))
            saved_results_oA['pred'].append(output_dict['pred_object_3d_afford'].cpu().numpy())
            saved_results_oA['gt'].append(input_dict['gt_contact_3d'].cpu().numpy())
            saved_results_oA['sim'].append(sim)
            saved_results_oA['mae'].append(mae)
            saved_results_oA['auc'].append(auc)
            saved_results_oA['iou'].append(iou)
            if valid_samples == 0:
                print(f'No valid samples for object contact metrics in batch {batch_idx}, skipping...')
                continue
            oafford_sim_meter.update(sim)
            oafford_mae_meter.update(mae)
            oafford_auc_meter.update(auc)
            oafford_iou_meter.update(iou)  

        # object contact metrics
        if ocontact_metric:
            f1, precision, recall = get_o_contact_metrics(input_dict['gt_contact_3d'], output_dict['pred_object_3d_contact'])
            ocontact_f1_meter.update(f1)
            ocontact_precision_meter.update(precision)
            ocontact_recall_meter.update(recall)

            # save results
            saved_results_oC['imgnames'].append(input_dict['image_paths'])
            saved_results_oC['pred'].append(output_dict['pred_object_3d_contact'].cpu().numpy())
            saved_results_oC['gt'].append(input_dict['gt_contact_3d'].cpu().numpy())
            saved_results_oC['f1'].append(f1)

        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou)

        if args.local_rank == 0:
            if np.random.rand() > 0.7:
                log_step = epoch * len(val_loader) + batch_idx
                log_images(loggers, 0, f'val-{ds_name}/images', input_dict, output_dict, log_step, args.disp_size)

    print(f'Finished validation for {ds_name} dataset with {len(val_loader)} batches and with batch_idx: {batch_idx}')

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    if hcontact_metric:
        hcontact_f1_meter.all_reduce()
        hcontact_precision_meter.all_reduce()
        hcontact_recall_meter.all_reduce()
        hcontact_geo_meter.all_reduce()

        saved_results_hC['pred'] = np.vstack(saved_results_hC['pred'])
        saved_results_hC['gt'] = np.vstack(saved_results_hC['gt'])
        saved_results_hC['avg_f1'] = hcontact_f1_meter.avg
        saved_results_hC['avg_precision'] = hcontact_precision_meter.avg
        saved_results_hC['avg_recall'] = hcontact_recall_meter.avg
        saved_results_hC['avg_geo'] = hcontact_geo_meter.avg

    if oafford_metric:
        oafford_sim_meter.all_reduce()
        oafford_mae_meter.all_reduce()
        oafford_auc_meter.all_reduce()
        oafford_iou_meter.all_reduce()

        saved_results_oA['pred'] = np.vstack(saved_results_oA['pred'])
        saved_results_oA['gt'] = np.vstack(saved_results_oA['gt'])
        saved_results_oA['avg_sim'] = oafford_sim_meter.avg
        saved_results_oA['avg_mae'] = oafford_mae_meter.avg
        saved_results_oA['avg_auc'] = oafford_auc_meter.avg
        saved_results_oA['avg_iou'] = oafford_iou_meter.avg

    if ocontact_metric:
        ocontact_f1_meter.all_reduce()
        ocontact_precision_meter.all_reduce()
        ocontact_recall_meter.all_reduce()

        saved_results_oC['avg_f1'] = ocontact_f1_meter.avg
        saved_results_oC['avg_precision'] = ocontact_precision_meter.avg
        saved_results_oC['avg_recall'] = ocontact_recall_meter.avg

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:

        log_dict = {
            f"val-{ds_name}/giou": giou, 
            f"val-{ds_name}/ciou": ciou, 
            "epoch": epoch
        }
        if hcontact_metric:
            log_dict.update({
                f"val-{ds_name}/C_f1": hcontact_f1_meter.avg,
                f"val-{ds_name}/C_precision": hcontact_precision_meter.avg,
                f"val-{ds_name}/C_recall": hcontact_recall_meter.avg,
                f"val-{ds_name}/C_geo": hcontact_geo_meter.avg,
            })
        if oafford_metric:
            log_dict.update({
                f"val-{ds_name}/O_sim": oafford_sim_meter.avg,
                f"val-{ds_name}/O_mae": oafford_mae_meter.avg,
                f"val-{ds_name}/O_auc": oafford_auc_meter.avg,
                f"val-{ds_name}/O_iou": oafford_iou_meter.avg,
            })
        if ocontact_metric:
            log_dict.update({
                f"val-{ds_name}/C_f1": ocontact_f1_meter.avg,
                f"val-{ds_name}/C_precision": ocontact_precision_meter.avg,
                f"val-{ds_name}/C_recall": ocontact_recall_meter.avg,
            })
        log_metric(loggers, log_dict, epoch)
        
        print(f'Epoch: {epoch}, {ds_name}: giou: {giou:.4f}, ciou: {ciou:.4f}')
        if hcontact_metric:
            print(f'Contact Metrics (Semantic) for human - F1: {hcontact_f1_meter.avg:.4f}, '
                  f'Precision: {hcontact_precision_meter.avg:.4f}, '
                  f'Recall: {hcontact_recall_meter.avg:.4f}',
                  f'Geo: {hcontact_geo_meter.avg:.4f}')
        if oafford_metric:
            print(f'Contact Metrics for objects - Sim: {oafford_sim_meter.avg:.4f}, '
                  f'MAE: {oafford_mae_meter.avg:.4f}, '
                  f'AUC: {oafford_auc_meter.avg:.4f}',
                  f'IoU: {oafford_iou_meter.avg:.4f}')
        if ocontact_metric:
            print(f'Contact Metrics for objects - F1: {ocontact_f1_meter.avg:.4f}, '
                  f'Precision: {ocontact_precision_meter.avg:.4f}, '
                  f'Recall: {ocontact_recall_meter.avg:.4f}')

    if hcontact_metric:
        return saved_results_hC, giou, ciou, hcontact_f1_meter.avg
    elif oafford_metric:
        return saved_results_oA, giou, ciou, oafford_iou_meter.avg
    elif ocontact_metric:
        return saved_results_oC, giou, ciou, ocontact_f1_meter.avg
    else:
        return None, giou, ciou, 0.0

def get_validation_dataset(args, tokenizer):
    
    contact_dataset_config = {
        "oC_sam_view_type": getattr(args, "oC_sam_view_type", None), 
        "oC_sam_input_type": getattr(args, "oC_sam_input_type", None),
        "oC_question_type": getattr(args, "oC_question_type", None), 
        "oC_ranking": getattr(args, "oC_ranking", None),  
        "hC_sam_view_type": getattr(args, "hC_sam_view_type", None), 
        "hC_sam_input_type": getattr(args, "hC_sam_input_type", None),
        "hC_mask_type": getattr(args, "hC_mask_type", None), 
        "hC_question_type": getattr(args, "hC_question_type", None),
        "token_type": getattr(args, "token_type", None),
    }
    val_datasets_map = {}
    for val_ds_name_str in args.val_dataset.split("||"):
        current_ds = None
        img_size_to_use = getattr(args, 'image_size', 1024) # from loaded config
        vision_tower_path = getattr(args, 'vision_tower') # from loaded config

        if 'ReasonSeg' in val_ds_name_str:
            current_ds = ValDataset(args.dataset_dir, tokenizer, vision_tower_path, val_ds_name_str, img_size_to_use)
        elif 'hcontact' in val_ds_name_str:
            current_ds = HContactSegDataset(args.dataset_dir, tokenizer, vision_tower_path, contact_dataset_config, is_train=False, image_size=img_size_to_use, val_dataset=val_ds_name_str)
        elif 'oafford' in val_ds_name_str:
            current_ds = OAffordSegDataset(args.dataset_dir, tokenizer, vision_tower_path, contact_dataset_config, is_train=False, image_size=img_size_to_use, val_dataset=val_ds_name_str)
        elif 'ocontact' in val_ds_name_str:
            current_ds = OContactSegDataset(args.dataset_dir, tokenizer, vision_tower_path, contact_dataset_config, is_train=False, image_size=img_size_to_use, val_dataset=val_ds_name_str)
        
        if current_ds:
            val_datasets_map[val_ds_name_str] = current_ds
            if args.local_rank == 0: print(f'Loaded Eval Dataset: {val_ds_name_str} with size: {len(current_ds)}')
        else:
            if args.local_rank == 0: print(f"Warning: Unknown or unhandled validation dataset type: {val_ds_name_str}. Skipping.")

    return val_datasets_map          

def get_validation_dataloader(args, val_datasets_map, tokenizer):
    val_loaders = {}
    conv_type_to_use = getattr(args, 'conv_type', 'llava_v1') # from loaded config
    use_mm_start_end_to_use = getattr(args, 'use_mm_start_end', True) # from loaded config

    for val_ds_name, dataset_obj in val_datasets_map.items():
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_obj, shuffle=False, drop_last=False) if torch.distributed.is_initialized() else None
        val_loaders[val_ds_name] = torch.utils.data.DataLoader(
            dataset_obj, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers,
            pin_memory=True, sampler=val_sampler,
            collate_fn=partial(collate_fn, tokenizer=tokenizer, conv_type=conv_type_to_use, 
                               use_mm_start_end=use_mm_start_end_to_use)
        )
    return val_loaders

def get_damon_semantic_contact(saved_results_hC):
    
    #### Object-wise semantic contact metric #####
    objnames_flat = [obj[0][0].lower() for obj in saved_results_hC['objnames']]
    results_by_object = {}
    for i, obj in enumerate(objnames_flat):
        results_by_object.setdefault(obj, []).append(i)

    semantic_results = {}
    for obj, indices in results_by_object.items():
        preds = [saved_results_hC['pred'][i] for i in indices]
        gts = [saved_results_hC['gt'][i] for i in indices]
        geo = [saved_results_hC['geo'][i] for i in indices]
        f1s = [saved_results_hC['f1'][i] for i in indices]

        tp = sum(np.sum(np.logical_and(p, g)) for p, g in zip(preds, gts))
        pred_pos = sum(np.sum(p) for p in preds)
        gt_pos = sum(np.sum(g) for g in gts)

        precision = tp / pred_pos if pred_pos > 0 else 0
        recall = tp / gt_pos if gt_pos > 0 else 0
        f1 = np.mean(f1s)
        geo_avg = np.mean(geo)

        semantic_results[obj] = {
            'num_samples': len(indices), 'avg_f1': f1, 'precision': precision,
            'recall': recall, 'geo': geo_avg
        }

    # Weighted averages
    total_samples = sum(r['num_samples'] for r in semantic_results.values())
    weighted_f1 = sum(r['avg_f1'] * r['num_samples'] for r in semantic_results.values()) / total_samples
    weighted_geo = sum(r['geo'] * r['num_samples'] for r in semantic_results.values()) / total_samples

    # Save or log if needed
    print(f"\n[DAMON-HCONTACT - Semantic Contact]")
    print(f"Weighted F1: {weighted_f1:.4f}, Weighted Geo: {weighted_geo:.4f}")

    #### High-level category wise contact metric #####
    category_metrics = {}
    for category, obj_list in DAMON_CATEGORIES_MAPPING.items():
        indices = [i for i, obj in enumerate(objnames_flat) if obj in obj_list]
        if not indices:
            continue

        preds = [saved_results_hC['pred'][i] for i in indices]
        gts = [saved_results_hC['gt'][i] for i in indices]
        geo = [saved_results_hC['geo'][i] for i in indices]
        f1s = [saved_results_hC['f1'][i] for i in indices]

        tp = sum(np.sum(np.logical_and(p, g)) for p, g in zip(preds, gts))
        pred_pos = sum(np.sum(p) for p in preds)
        gt_pos = sum(np.sum(g) for g in gts)

        precision = tp / pred_pos if pred_pos > 0 else 0
        recall = tp / gt_pos if gt_pos > 0 else 0
        f1 = np.mean(f1s)
        geo_avg = np.mean(geo)

        category_metrics[category] = {
            'num_samples': len(indices), 'avg_f1': f1,
            'precision': precision, 'recall': recall, 'geo': geo_avg
        }

    # Print category-level summary
    print(f"\n[DAMON-HCONTACT - Semantic Contact Category Summary]")
    print(f"{'Category':20} | {'Samples':>7} | {'F1':>6} | {'Prec':>6} | {'Recall':>6} | {'Geo':>6}")
    print("-" * 70)
    for cat, m in category_metrics.items():
        print(f"{cat:20} | {m['num_samples']:7d} | {m['avg_f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['geo']:.4f}")


def get_damon_binary_contact(saved_results_hC):
    # Binary contact
    imgwise = {}
    for i, name in enumerate(saved_results_hC['imgnames']):
        key = name[0]  # image name
        if key not in imgwise:
            imgwise[key] = {
                'pred': saved_results_hC['pred'][i],
                'gt': saved_results_hC['gt'][i],
                'geo': saved_results_hC['geo'][i]
            }
        else:
            imgwise[key]['pred'] = np.logical_or(imgwise[key]['pred'], saved_results_hC['pred'][i])
            imgwise[key]['gt'] = np.logical_or(imgwise[key]['gt'], saved_results_hC['gt'][i])
            imgwise[key]['geo'] = max(imgwise[key]['geo'], saved_results_hC['geo'][i])

    f1_scores, geos = [], []
    tp, pred_pos, gt_pos = 0, 0, 0
    for v in imgwise.values():
        p, g = v['pred'], v['gt']
        tpi = np.sum(np.logical_and(p, g))
        ppi = np.sum(p)
        gpi = np.sum(g)
        prec = tpi / ppi if ppi else 0
        rec = tpi / gpi if gpi else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        f1_scores.append(f1)
        geos.append(v['geo'])

        tp += tpi
        pred_pos += ppi
        gt_pos += gpi

    global_precision = tp / pred_pos if pred_pos else 0
    global_recall = tp / gt_pos if gt_pos else 0
    global_f1 = np.mean(f1_scores)
    global_geo = np.mean(geos)

    print(f"\n[DAMON-HCONTACT - Binary Contact]")
    print(f"Global F1: {global_f1:.4f}, Precision: {global_precision:.4f}, Recall: {global_recall:.4f}, Geo: {global_geo:.4f}")


def parse_args(args):
    parser = argparse.ArgumentParser(description="InteractVLM Model Evaluation")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument('--eval_only', default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument("--disp_size", default=128, type=int)
    parser.add_argument("--val_dataset", default="piad_oafford", type=str)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument('--log_wandb', default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    return parser.parse_args(args)

def main_eval(args):
    
    args = parse_args(args)
    args = get_args_for_eval(args)

    print(f'disp_size: {args.disp_size}')
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    print(f'All the arguments: {args}')

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, cache_dir=None, model_max_length=args.model_max_length,
        padding_side="right", use_fast=False, legacy=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer, args = add_new_tokens(tokenizer, args) 
    if args.use_mm_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    
    # Determine multiview_channels based on loaded config (as before)
    mv_human, mv_obj = 1, 1
    if args.train_from_LISA or args.train_from_LLAVA:
        if 'MV-Z' in args.hC_sam_view_type and 'hcontact' in args.dataset:
            mv_human = HUMAN_VIEW_DICT[args.hC_sam_view_type]['grid_size'][0]
        if 'MV-Z' in args.oC_sam_view_type and ('oafford' in args.dataset or 'ocontact' in args.dataset):
            mv_obj = OBJS_VIEW_DICT[args.oC_sam_view_type]['grid_size'][0]
        args.multiview_channels = int(max(mv_human, mv_obj))

    loggers = [None, None]
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        copy_code(args.log_dir) # copy code to log dir
        with open(os.path.join(args.log_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        writer = SummaryWriter(args.log_dir)
        loggers[0] = writer
        if args.log_wandb:
            wandb_logger = wandb.init(project="CHARM",
                                    name=args.exp_name, 
                                    dir=args.log_dir,
                                    config=vars(args))
            loggers[1] = wandb_logger
    
    # check if precision is bf16 else raise error
    assert args.precision == "bf16",  "Only bf16 is supported for inference."
    torch_dtype = torch.bfloat16
    
    kwargs = {"torch_dtype": torch_dtype}
    kwargs.update({"train_from_LISA": False})
    kwargs.update({"train_from_LLAVA": False})


    model = InteractVLMForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    model = model.bfloat16().cuda()
    model.eval()

    use_different_decoders = 'DifDe' in args.token_type
    if (args.train_from_LISA or args.train_from_LLAVA) and use_different_decoders:
        model.get_model().initialize_separate_decoders()

    if getattr(args, 'train_from_LLAVA', False): # From loaded training config
        model.get_model().initialize_ivlm_modules(model.get_model().config)
    if (getattr(args, 'train_from_LISA', True) or getattr(args, 'train_from_LLAVA', False)) and \
       'DifDe' in getattr(args, 'token_type', ''): # From loaded training config
        model.get_model().initialize_separate_decoders()
    
    model.resize_token_embeddings(len(tokenizer))
    
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config={
            "train_micro_batch_size_per_gpu": args.val_batch_size,
            "bf16": {"enabled": args.precision == "bf16"},
        }
    )

    # Setup validation datasets and dataloaders
    val_datasets_map = get_validation_dataset(args, tokenizer)
    val_loaders = get_validation_dataloader(args, val_datasets_map, tokenizer)

    if args.local_rank == 0:
        print(f'Validation datasets: {val_datasets_map.keys()}')
        print(f'Validation dataloaders: {val_loaders.keys()}')

    conv_type_to_use = getattr(args, 'conv_type', 'llava_v1') # from loaded config
    conversation_lib.default_conversation = conversation_lib.conv_templates[conv_type_to_use]
    all_results_summary = {}

    for val_ds_name_key, val_loader_instance in val_loaders.items():
        print(f"\n--- Evaluating on {val_ds_name_key} ---")
        saved_results, giou, ciou, contact_metric = validate(
            val_loader_instance, model_engine, 0, loggers, args, val_ds_name_key
        )
        summary = {'giou': giou, 'ciou': ciou, 'contact_metric': contact_metric}
        all_results_summary[val_ds_name_key] = summary
        if args.local_rank == 0 and saved_results is not None:
            results_path = os.path.join(args.log_dir, f'{val_ds_name_key}_results.pkl')
            jl.dump(saved_results, results_path)
            print(f"Saved detailed results for {val_ds_name_key} to {results_path}")

        if val_ds_name_key == 'damon_hcontact':
            get_damon_semantic_contact(saved_results)
            get_damon_binary_contact(saved_results)


if __name__ == "__main__":
    main_eval(sys.argv[1:])