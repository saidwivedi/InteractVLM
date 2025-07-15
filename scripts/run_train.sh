#!/bin/bash

# Set environment variables
export PATH=$PATH
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
export HF_HOME='.cache/huggingface'
export HF_HOME='/is/cluster/sdwivedi/.cache/huggingface'


# generate random number between 20000 and 25000
MASTER_PORT=$(( ( RANDOM % 5000 )  + 20000 ))
VISION_PRETRAINED="./data/sam_vit_h_4b8939.pth"


EPOCHS='30'
BATCH_SIZE='16'
LOG_WANDB="True"

PRINT_FREQ='10'
DISP_FREQ="100"
DISP_SIZE="128"

IMG_EMB_LEN="255"
TRAIN_FROM_LISA="True"
TRAIN_FROM_LLAVA="False"
GRAD_ACCUMULATION_STEPS="1"
STEPS_PER_EPOCH="500"
EVAL_ONLY="False"
NO_EVAL="False"

LORA_R="8"

OAFFORD_SEG_DATA="piad_oafford||lemon_oafford"
OCONTACT_SEG_DATA="pico_ocontact"
HCONTACT_SEG_DATA="damon_hcontact||lemon_hcontact"
HCONTACTSCENE_SEG_DATA="rich_hcontact"
VQA_DATA="llava"
VAL_DATASET="piad_oafford"

OC_SAM_INPUT_TYPE="color"
OC_SAM_VIEW_TYPE="4MV-Z_HM"
OC_RANKING="lookup"
HC_SAM_INPUT_TYPE="norm"
HC_SAM_VIEW_TYPE="4MV-Z_Vitru"
HC_MASK_TYPE="objectwise"
OC_QUESTION_TYPE="afford_obj"
HC_QUESTION_TYPE="parts"
HC_TRAIN_FRACTION="1.0"

HC_LOSS_WEIGHT="0.0"
OC_LOSS_WEIGHT="0.0"
BCE_LOSS_WEIGHT="0.0"
BCE_LOSS_ALPHA="0.0"
DICE_LOSS_SCALE="1"
DICE_LOSS_WEIGHT="0.0"

VERSION="xinlai/LISA-13B-llama2-v1"
CLIP_MODEL="openai/clip-vit-large-patch14"

USE_FEAT_FUSION="False"
USE_UNCERTAINTY="False"
TOKEN_TYPE="Gen"
MULTIVIEW_CAM_COND="False"
CAM_ENCODER_TYPE="simple"


DATASET_DIR="./data"

# Function to set configuration based on numerical argument
set_configuration() {
  case $1 in
    "hcontact-damon")
      EXP_NAME="interactvlm-3d-hcontact-damon"
      DATASET="hcontact_seg"
      SAMPLE_RATES="1"
      HC_SAM_INPUT_TYPE="norm"
      HC_SAM_VIEW_TYPE="4MV-Z_Vitru"
      HC_MASK_TYPE="objectwise"
      HC_QUESTION_TYPE="parts"
      HC_TRAIN_FRACTION="1.0"
      BATCH_SIZE="8"
      GRAD_ACCUMULATION_STEPS="1"
      HCONTACT_SEG_DATA="damon_hcontact"
      VAL_DATASET="damon_hcontact"
      USE_FEAT_FUSION="False"
      USE_UNCERTAINTY="False"
      MULTIVIEW_CAM_COND="True"
      CAM_ENCODER_TYPE="vi_v1"
      HC_LOSS_WEIGHT="3.0"
      BCE_LOSS_WEIGHT="2.0"
      BCE_LOSS_ALPHA="0.5"
      DICE_LOSS_SCALE="1.0"
      DICE_LOSS_WEIGHT="1.0"
      TOKEN_TYPE="Gen"
      TRAIN_FROM_LISA="True"
      TRAIN_FROM_LLAVA="False"
      IMG_EMB_LEN="255"
      LOG_WANDB="True"
      ;;
    "hcontact-wScene")
      EXP_NAME="interactvlm-3d-hcontact-wScene"
      DATASET="hcontact_seg||hcontactScene_seg||vqa"
      SAMPLE_RATES="9,3,1"
      OC_SAM_VIEW_TYPE="4MV-Z_HM"
      OC_SAM_INPUT_TYPE="color"
      OC_RANKING="lookup"
      HC_SAM_INPUT_TYPE="norm"
      HC_SAM_VIEW_TYPE="4MV-Z_Vitru_wScene"
      HC_MASK_TYPE="objectwise"
      OC_QUESTION_TYPE="afford"
      HC_QUESTION_TYPE="parts"
      HC_TRAIN_FRACTION="1.0"
      BATCH_SIZE="8"
      GRAD_ACCUMULATION_STEPS="1"
      OAFFORD_SEG_DATA="lemon_oafford"
      HCONTACT_SEG_DATA="damon_hcontact||lemon_hcontact"
      HCONTACTSCENE_SEG_DATA="rich_hcontact"
      VQA_DATA="llava||damon||lemon"
      VAL_DATASET="damon_hcontact||lemon_hcontact"
      USE_FEAT_FUSION="False"
      USE_UNCERTAINTY="False"
      MULTIVIEW_CAM_COND="True"
      CAM_ENCODER_TYPE="vi_v1"
      HC_LOSS_WEIGHT="3.0"
      OC_LOSS_WEIGHT="0.0"
      BCE_LOSS_WEIGHT="2.0"
      BCE_LOSS_ALPHA="0.5"
      DICE_LOSS_SCALE="1.0"
      DICE_LOSS_WEIGHT="1.0"
      TOKEN_TYPE="Gen"
      TRAIN_FROM_LISA="True"
      TRAIN_FROM_LLAVA="False"
      IMG_EMB_LEN="255"
      LOG_WANDB="True"
      ;;
    "oafford-lemon-piad")
      EXP_NAME="interactvlm-3d-oafford-lemon-piad"
      DATASET="offord_seg"
      SAMPLE_RATES="1"
      OC_SAM_VIEW_TYPE="4MV-Z_HM"
      OC_SAM_INPUT_TYPE="color"
      OC_RANKING="lookup"
      OC_QUESTION_TYPE="afford"
      BATCH_SIZE="8"
      GRAD_ACCUMULATION_STEPS="1"
      OAFFORD_SEG_DATA="piad_oafford||lemon_oafford"
      VAL_DATASET="piad_oafford"
      USE_FEAT_FUSION="False"
      USE_UNCERTAINTY="False"
      MULTIVIEW_CAM_COND="True"
      CAM_ENCODER_TYPE="vi_v1"
      OC_LOSS_WEIGHT="3.0"
      BCE_LOSS_WEIGHT="2.0"
      BCE_LOSS_ALPHA="0.5"
      DICE_LOSS_SCALE="1.0"
      DICE_LOSS_WEIGHT="1.0"
      TOKEN_TYPE="Gen"
      TRAIN_FROM_LISA="True"
      TRAIN_FROM_LLAVA="False"
      IMG_EMB_LEN="255"
      LOG_WANDB="True"
      ;;
    *)
      echo "Unknown configuration number: $1"
      exit 1
      ;;
  esac
}

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Please provide a configuration number (e.g., 0, 1)."
  exit 1
fi

# Set the configuration based on the numerical argument
set_configuration $1


# Run the DeepSpeed training
echo "deepspeed \
  --master_port=$MASTER_PORT train.py \
  --version=$VERSION \
  --vision_pretrained=$VISION_PRETRAINED \
  --vision-tower=$CLIP_MODEL \
  --img_emb_len=$IMG_EMB_LEN \
  --lora_r=$LORA_R \
  --dataset=$DATASET \
  --val_dataset=$VAL_DATASET \
  --train_from_LISA=$TRAIN_FROM_LISA \
  --train_from_LLAVA=$TRAIN_FROM_LLAVA \
  --eval_only=$EVAL_ONLY \
  --no_eval=$NO_EVAL \
  --oafford_seg_data=$OAFFORD_SEG_DATA \
  --ocontact_seg_data=$OCONTACT_SEG_DATA \
  --hcontact_seg_data=$HCONTACT_SEG_DATA \
  --hcontactScene_seg_data=$HCONTACTSCENE_SEG_DATA \
  --vqa_data=$VQA_DATA \
  --oC_sam_view_type=$OC_SAM_VIEW_TYPE \
  --oC_sam_input_type=$OC_SAM_INPUT_TYPE \
  --oC_ranking=$OC_RANKING \
  --hC_sam_view_type=$HC_SAM_VIEW_TYPE \
  --hC_sam_input_type=$HC_SAM_INPUT_TYPE \
  --oC_question_type=$OC_QUESTION_TYPE \
  --hC_question_type=$HC_QUESTION_TYPE \
  --hC_train_fraction=$HC_TRAIN_FRACTION \
  --hC_loss_weight=$HC_LOSS_WEIGHT \
  --oC_loss_weight=$OC_LOSS_WEIGHT \
  --bce_loss_weight=$BCE_LOSS_WEIGHT \
  --dice_loss_weight=$DICE_LOSS_WEIGHT \
  --bce_loss_alpha=$BCE_LOSS_ALPHA \
  --dice_loss_scale=$DICE_LOSS_SCALE \
  --hC_mask_type=$HC_MASK_TYPE \
  --sample_rates=$SAMPLE_RATES \
  --exp_name=$EXP_NAME \
  --epochs=$EPOCHS \
  --batch_size=$BATCH_SIZE \
  --display_freq=$DISP_FREQ \
  --disp_size=$DISP_SIZE \
  --print_freq=$PRINT_FREQ \
  --log_wandb=$LOG_WANDB \
  --grad_accumulation_steps=$GRAD_ACCUMULATION_STEPS \
  --steps_per_epoch=$STEPS_PER_EPOCH \
  --use_feat_fusion=$USE_FEAT_FUSION \
  --use_uncertainty=$USE_UNCERTAINTY \
  --token_type=$TOKEN_TYPE \
  --multiview_cam_cond=$MULTIVIEW_CAM_COND \
  --cam_encoder_type=$CAM_ENCODER_TYPE"


deepspeed \
  --master_port=$MASTER_PORT train.py \
  --version=$VERSION \
  --vision_pretrained=$VISION_PRETRAINED \
  --vision-tower=$CLIP_MODEL \
  --img_emb_len=$IMG_EMB_LEN \
  --lora_r=$LORA_R \
  --dataset=$DATASET \
  --val_dataset=$VAL_DATASET \
  --train_from_LISA=$TRAIN_FROM_LISA \
  --train_from_LLAVA=$TRAIN_FROM_LLAVA \
  --eval_only=$EVAL_ONLY \
  --no_eval=$NO_EVAL \
  --oafford_seg_data=$OAFFORD_SEG_DATA \
  --ocontact_seg_data=$OCONTACT_SEG_DATA \
  --hcontact_seg_data=$HCONTACT_SEG_DATA \
  --hcontactScene_seg_data=$HCONTACTSCENE_SEG_DATA \
  --vqa_data=$VQA_DATA \
  --oC_sam_view_type=$OC_SAM_VIEW_TYPE \
  --oC_sam_input_type=$OC_SAM_INPUT_TYPE \
  --oC_ranking=$OC_RANKING \
  --hC_sam_view_type=$HC_SAM_VIEW_TYPE \
  --hC_sam_input_type=$HC_SAM_INPUT_TYPE \
  --oC_question_type=$OC_QUESTION_TYPE \
  --hC_question_type=$HC_QUESTION_TYPE \
  --hC_train_fraction=$HC_TRAIN_FRACTION \
  --hC_loss_weight=$HC_LOSS_WEIGHT \
  --oC_loss_weight=$OC_LOSS_WEIGHT \
  --bce_loss_weight=$BCE_LOSS_WEIGHT \
  --dice_loss_weight=$DICE_LOSS_WEIGHT \
  --bce_loss_alpha=$BCE_LOSS_ALPHA \
  --dice_loss_scale=$DICE_LOSS_SCALE \
  --hC_mask_type=$HC_MASK_TYPE \
  --sample_rates=$SAMPLE_RATES \
  --exp_name=$EXP_NAME \
  --epochs=$EPOCHS \
  --batch_size=$BATCH_SIZE \
  --display_freq=$DISP_FREQ \
  --disp_size=$DISP_SIZE \
  --print_freq=$PRINT_FREQ \
  --log_wandb=$LOG_WANDB \
  --grad_accumulation_steps=$GRAD_ACCUMULATION_STEPS \
  --steps_per_epoch=$STEPS_PER_EPOCH \
  --use_feat_fusion=$USE_FEAT_FUSION \
  --use_uncertainty=$USE_UNCERTAINTY \
  --token_type=$TOKEN_TYPE \
  --multiview_cam_cond=$MULTIVIEW_CAM_COND \
  --cam_encoder_type=$CAM_ENCODER_TYPE
