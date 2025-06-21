import argparse
import os
import shutil
import sys
import time
import joblib as jl
from functools import partial
from distutils.util import strtobool

import deepspeed
import numpy as np
import torch
import tqdm
import json
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
import wandb

from model.InteractVLM import InteractVLMForCausalLM
from model.llava import conversation as conversation_lib
from datasets.dataset import HybridDataset, collate_fn
from utils.utils import (get_dtype, dict_to_cuda, copy_code, add_new_tokens, log_images, log_metric,
                         AverageMeter, ProgressMeter)
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN 
from preprocess_data.constants import HUMAN_VIEW_DICT, OBJS_VIEW_DICT
from evaluate import get_validation_dataset, get_validation_dataloader, validate


def parse_args(args):
    parser = argparse.ArgumentParser(description="InteractVLM Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--img_emb_len", default=255, type=int, help="255 for llava and 575 for llava1.5")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", 
        default="sem_seg||refer_seg||vqa||reason_seg||hcontact_seg||oafford_seg||ocontact_seg||h2dcontact_seg", 
        type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava||damon", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--oafford_seg_data", default="piad_oafford||lemon_oafford", type=str)
    parser.add_argument("--ocontact_seg_data", default="pico_ocontact", type=str)
    parser.add_argument("--hcontact_seg_data", default="damon_hcontact||lemon_hcontact", type=str)
    parser.add_argument("--h2dcontact_seg_data", default="damon_h2dcontact", type=str)
    parser.add_argument("--val_dataset", default="damon_hcontact", type=str)
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="interactvlm", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--hC_loss_weight", default=3.0, type=float)
    parser.add_argument("--oC_loss_weight", default=3.0, type=float)
    parser.add_argument("--bce_loss_alpha", default=0.25, type=float)
    parser.add_argument("--dice_loss_scale", default=1000, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument('--eval_only', default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--no_eval', default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument("--vision_pretrained", default="../pretrained_models/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--display_freq", default=200, type=int)
    parser.add_argument("--disp_size", default=128, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    # contact dataset config
    parser.add_argument("--oC_sam_view_type", default='4viewsFix', type=str)
    parser.add_argument("--oC_sam_input_type", default='color', type=str)
    parser.add_argument("--oC_ranking", default='random', type=str)
    parser.add_argument("--oC_question_type", default='simple', type=str)
    parser.add_argument("--hC_sam_view_type", default='2viewsFix', type=str)
    parser.add_argument("--hC_sam_input_type", default='norm', type=str)
    parser.add_argument("--hC_mask_type", default='all_contact', type=str)
    parser.add_argument("--hC_question_type", default='simple', type=str)
    parser.add_argument("--hC_train_fraction", default=1.0, type=float)
    # newly added configs
    parser.add_argument('--train_from_LISA', default=True, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--train_from_LLAVA', default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--token_type', default='Gen', type=str, help='Gen or Gen-Hu-Obj or Gen-Int')
    parser.add_argument('--use_feat_fusion', default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--use_uncertainty', default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--multiview_cam_cond', default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--cam_encoder_type', default='simple', type=str)
    parser.add_argument('--log_wandb', default=True, type=lambda x: bool(strtobool(x)))
    return parser.parse_args(args)


def main(args):

    args = parse_args(args)

    if args.eval_only:
        print(f"Run evaluate.py for evaluation")
        exit()

    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    print(f'All the arguments: {args}')

    legacy = True ## TODO: check why this needs to be true
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        legacy=legacy,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Add Segmentation Tokens
    tokenizer, args = add_new_tokens(tokenizer, args)

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    
    # Check how many multi-view channels are required, default is 1
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
            wandb_logger = wandb.init(project="interactvlm",
                                    name=args.exp_name, 
                                    dir=args.log_dir,
                                    config=vars(args))
            loggers[1] = wandb_logger
            
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "img_emb_len": args.img_emb_len,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "dice_loss_scale": args.dice_loss_scale,
        "bce_loss_weight": args.bce_loss_weight,
        "bce_loss_alpha": args.bce_loss_alpha,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "train_from_LISA": args.train_from_LISA,
        "train_from_LLAVA": args.train_from_LLAVA,
        "use_feat_fusion": args.use_feat_fusion,
        "use_uncertainty": args.use_uncertainty,
        "token_type": args.token_type,
        "hseg_token_idx": args.hseg_token_idx,
        "oseg_token_idx": args.oseg_token_idx,
        "hC_sam_view_type": args.hC_sam_view_type,
        "hC_loss_weight": args.hC_loss_weight,
        "hC_question_type": args.hC_question_type,
        "oC_sam_view_type": args.oC_sam_view_type,
        "oC_loss_weight": args.oC_loss_weight,
        "oC_question_type": args.oC_question_type,
        "multiview_channels": args.multiview_channels,
        "multiview_cam_cond": args.multiview_cam_cond,
        "cam_encoder_type": args.cam_encoder_type,
    }
    contact_dataset_config = {
        "oC_sam_view_type": args.oC_sam_view_type,
        "oC_sam_input_type": args.oC_sam_input_type,
        "oC_question_type": args.oC_question_type,
        "oC_ranking": args.oC_ranking,  
        "hC_sam_view_type": args.hC_sam_view_type,
        "hC_sam_input_type": args.hC_sam_input_type,
        "hC_mask_type": args.hC_mask_type,
        "hC_question_type": args.hC_question_type,
        "hC_train_fraction": args.hC_train_fraction,
        "token_type": args.token_type,
    }

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = InteractVLMForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, local_files_only=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)

    if args.train_from_LLAVA:
        model.get_model().initialize_ivlm_modules(model.get_model().config)

    # Since huggingface model are loaded internally, for Pretrained lisa models, we need to initialize the decoders outside
    use_different_decoders = 'DifDe' in args.token_type
    if (args.train_from_LISA or args.train_from_LLAVA) and use_different_decoders:
        model.get_model().initialize_separate_decoders()

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls) and all(x not in name for x in \
                    ["visual_model", "vision_tower", "mm_projector", "text_hidden_fcs"]) \
                    and any(x in name for x in lora_target_modules):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(x in n for x in ["lm_head", "embed_tokens", "text_hidden_fcs", "mask_decoder", 
                                 "human_mask_decoder", "object_mask_decoder", "fusion", 
                                 "uncertainty", "attention_splitter", "cam_pose_encoder"]):
            if args.local_rank == 0: print(f"Parameter {n} (shape: {p.shape}) is trainable.")
            p.requires_grad = True

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        contact_dataset_config,
        samples_per_epoch=args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        oafford_seg_data=args.oafford_seg_data,
        ocontact_seg_data=args.ocontact_seg_data,
        hcontact_seg_data=args.hcontact_seg_data,
        h2dcontact_seg_data=args.h2dcontact_seg_data, 
        explanatory=args.explanatory,
    )

    if args.no_eval == False:
        val_datasets = get_validation_dataset(args, tokenizer)
        for val_ds, dataset in val_datasets.items():
            print(f'----> Val Dataset: {val_ds} with size: {len(dataset)}')

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": args.lr, "weight_decay": 0.0, "betas": (args.beta1, args.beta2),},
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {"total_num_steps": args.epochs * args.steps_per_epoch, "warmup_min_lr": 0, 
                       "warmup_max_lr": args.lr, "warmup_num_steps": 100, "warmup_type": "linear",},
        },
        "fp16": {"enabled": args.precision == "fp16",},
        "bf16": {"enabled": args.precision == "bf16",},
        "gradient_clipping": 1.0,
        "zero_optimization": {"stage": 2, "contiguous_gradients": True, "overlap_comm": True, "reduce_scatter": True,
                              "reduce_bucket_size": 5e8, "allgather_bucket_size": 5e8,},
    }

    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
            multiview_channels=args.multiview_channels,
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    val_loaders = {}
    if not args.no_eval:
        
        val_loaders = get_validation_dataloader(args, val_datasets, tokenizer)

        best_score = 0.0
        best_scores_all = {val_ds: 0.0 for val_ds in val_loaders}
        ciou_scores_all = {val_ds: 0.0 for val_ds in val_loaders}

    train_iter = iter(train_loader)
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            loggers,
            train_iter,
            args,
        )

        save_model = True
        if not args.no_eval:
            for val_ds, val_loader in val_loaders.items():
                saved_results, giou, ciou, contact_metric = validate(val_loader, model_engine, epoch, loggers, args, val_ds)
                if ('hcontact' in val_ds or 'oafford' in val_ds) and contact_metric > 0:
                    best_scores_all[val_ds] = max(contact_metric, best_scores_all[val_ds])
                else:
                    best_scores_all[val_ds] = max(giou, best_scores_all[val_ds])
                ciou_scores_all[val_ds] = max(ciou, ciou_scores_all[val_ds])

            # Save the model based on the best score of the first dataset
            imp_ds = args.val_dataset.split("||")[0]
            is_best = best_scores_all[imp_ds] > best_score
            best_score = max(best_scores_all[imp_ds], best_score)

            if ('hcontact' in imp_ds or 'oafford' in imp_ds) and not is_best:
                print(f"Skipping saving model -- Best score: {best_score}, current score: {best_scores_all[imp_ds]}")
                save_model = False

        if save_model:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0 and not args.no_eval:
                if saved_results is not None:
                    jl.dump(saved_results, f'{args.log_dir}/{imp_ds}_results.pkl')
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_best_score{:.3f}_ciou{:.3f}.pth".format(
                            best_score, ciou_scores_all[imp_ds]
                        ),
                    ),
                )
            if os.path.exists(save_dir) and args.local_rank == 0:
                shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    loggers,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_l2_losses = AverageMeter("MaskL2Loss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    hC_losses = AverageMeter("hCLoss", ":.4f")
    oA_losses = AverageMeter("oALoss", ":.4f")
    oC_losses = AverageMeter("oCLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
            mask_l2_losses,
            hC_losses,
            oA_losses,
            oC_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    global_step = epoch * args.steps_per_epoch
    for step in range(args.steps_per_epoch):
        accumulated_loss = 0
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            input_dict["images"] = input_dict["images"].to(dtype=get_dtype(args.precision))
            input_dict["images_clip"] = input_dict["images_clip"].to(dtype=get_dtype(args.precision))
            input_dict["cam_params"] = input_dict["cam_params"].to(dtype=get_dtype(args.precision))

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_l2_loss = output_dict["mask_l2_loss"]
            mask_loss = output_dict["mask_loss"]
            hC_loss = output_dict["hC_loss"]
            oA_loss = output_dict["oA_loss"]
            oC_loss = output_dict["oC_loss"]

            # skip if loss is nan
            if torch.isnan(loss) or torch.isinf(loss):
                print("Nan or Inf loss encountered, skipping...")
                print(f'batch: {i}, step: {step}, epoch: {epoch}')
                print(f'image_paths: {input_dict["img_path"]}')
                continue

            # Normalize the loss according to gradient accumulation steps
            loss = loss / args.grad_accumulation_steps
            accumulated_loss += loss.item()

            # Backward pass
            model.backward(loss)

            if (i + 1) % args.grad_accumulation_steps == 0:
                # Update model parameters
                model.step()
                
                # Update metrics
                loss_update_size = input_dict["images"].size(0)
                losses.update(accumulated_loss, loss_update_size)
                ce_losses.update(ce_loss.item(), loss_update_size)
                mask_bce_losses.update(mask_bce_loss.item(), loss_update_size)
                mask_dice_losses.update(mask_dice_loss.item(), loss_update_size)
                mask_l2_losses.update(mask_l2_loss.item(), loss_update_size)
                mask_losses.update(mask_loss.item(), loss_update_size)
                hC_losses.update(hC_loss.item(), loss_update_size)
                oA_losses.update(oA_loss.item(), loss_update_size)
                oC_losses.update(oC_loss.item(), loss_update_size)


                # Reset accumulated loss
                accumulated_loss = 0

        # Log images and metrics after each step (which now includes multiple grad accumulation steps)
        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_l2_losses.all_reduce()
                mask_losses.all_reduce()
                hC_losses.all_reduce()
                oA_losses.all_reduce()
                oC_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(step + 1)
                metric_dict = {
                    "train/loss": losses.avg,
                    "train/ce_loss": ce_losses.avg,
                    "train/mask_bce_loss": mask_bce_losses.avg,
                    "train/mask_dice_loss": mask_dice_losses.avg,
                    "train/mask_l2_loss": mask_l2_losses.avg,
                    "train/mask_loss": mask_losses.avg,
                    "train/hC_loss": hC_losses.avg,
                    "train/oA_loss": oA_losses.avg,
                    "train/oC_loss": oC_losses.avg,
                    "train/total_secs_per_batch": batch_time.avg,
                    "train/data_secs_per_batch": data_time.avg,
                    "global_step": global_step,
                }
                log_metric(loggers, metric_dict, global_step)
        
        if global_step % args.display_freq == 0 and args.local_rank == 0:
            log_images(loggers, 0, 'train/images', input_dict, output_dict, global_step, args.disp_size)

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_l2_losses.reset()
            mask_losses.reset()
            hC_losses.reset()
            oA_losses.reset()
            oC_losses.reset()
        
        global_step += 1

    curr_lr = scheduler.get_last_lr()
    if args.local_rank == 0:
        log_metric(loggers, {"lr": curr_lr[0]}, global_step)


if __name__ == "__main__":
    main(sys.argv[1:])
