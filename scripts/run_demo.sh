#!/bin/bash

export PATH=$PATH
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
export HF_HOME='.cache/huggingface'
export HF_HOME='/is/cluster/sdwivedi/.cache/huggingface'


# Check if user provided the contact type
if [ -z "$1" ]; then
    echo "Usage: $0 <contact_type>"
    echo "Please provide the contact type:"
    echo "  hcontact  - for human contact demo"
    echo "  oafford   - for object affordance demo"
    exit 1
fi

contact_type=$1

# Validate the contact type and set the appropriate model path
case $contact_type in
    "hcontact")
        model_path="./trained_models/interactvlm-3d-hcontact-damon"
        echo "Running human contact demo..."
        ;;
    "oafford")
        model_path="./trained_models/interactvlm-3d-oafford-lemon-piad"
        echo "Running object affordance demo..."
        ;;
    *)
        echo "Error: Invalid contact type '$contact_type'"
        echo "Valid options are: hcontact, oafford"
        exit 1
        ;;
esac

input_img_folder="./data/demo_samples"

echo "<<<<<<<----->>>>>>> Running $contact_type demo"
echo "Using model: $model_path"

python run_demo.py \
        --version="$model_path" \
        --img_folder=${input_img_folder} \
        --contact_type="$contact_type"