#!/bin/bash

export PATH=$PATH
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
export HF_HOME='.cache/huggingface'
export HF_HOME='/is/cluster/sdwivedi/.cache/huggingface'


# Check if user provided the contact type
if [ -z "$1" ]; then
    echo "Usage: $0 <contact_type> [input_img_folder] [input_mode]"
    echo "Please provide the contact type:"
    echo "  hcontact  - for human contact demo"
    echo "  oafford   - for object affordance demo"
    echo "Optional: provide input image folder path (default: ./data/demo_samples)"
    echo "Optional: provide input mode (default: folder)"
    echo "  folder   - folder-based samples (each sample in its own folder)"
    echo "  file     - file-based samples (human contact only, all samples as files in single folder)"
    echo "Note: Object contact always uses folder-based structure"
    exit 1
fi

contact_type=$1
input_img_folder=${2:-"./data/demo_samples"}
input_mode=${3:-"folder"}

# Validate the contact type and set the appropriate model path
case $contact_type in
    "hcontact")
        model_path="./trained_models/interactvlm-3d-hcontact-damon"
        echo "Running human contact demo..."
        ;;
    "hcontact-wScene")
        model_path="./trained_models/interactvlm-3d-hcontact-wScene-damon-lemon-rich"
        echo "Running human contact with objects and scene demo..."
        ;;
    "oafford")
        model_path="./trained_models/interactvlm-3d-oafford-lemon-piad"
        echo "Running object affordance demo..."
        ;;
    "hcontact-ocontact")
        model_path="./trained_models/interactvlm-3d-hcontact-ocontact"
        echo "Running joint human-object contact demo..."
        ;;
    "h2dcontact")
        model_path="./trained_models/interactvlm-2d-hcontact"
        echo "Running 2D human contact demo..."
        ;;
    *)
        echo "Error: Invalid contact type '$contact_type'"
        echo "Valid options are: hcontact, oafford"
        exit 1
        ;;
esac

echo "<<<<<<<----->>>>>>> Running $contact_type demo"
echo "Using model: $model_path"
echo "Using input folder: $input_img_folder"
echo "Using input mode: $input_mode"

python run_demo.py \
        --version="$model_path" \
        --img_folder=${input_img_folder} \
        --contact_type="$contact_type" \
        --input_mode="$input_mode"