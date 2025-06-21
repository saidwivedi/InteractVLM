#!/bin/bash

export PATH=$PATH
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
export HF_HOME='.cache/huggingface'

root_dir="/path/to/InteractVLM"


exps=("DAM_LLaVA-13B")
save_paths=("DAM_LLaVA-13B")

# Get the specific number from the command line argument
if [ -z "$1" ]; then
    echo "Please provide a number to select the experiment."
    exit 1
fi

i=$1

if [ $i -ge ${#exps[@]} ] || [ $i -lt 0 ]; then
    echo "Invalid number. Please select a number between 0 and $((${#exps[@]} - 1))."
    exit 1
fi

exp=${exps[$i]}
save_path=${save_paths[$i]}
echo "Selected experiment: $exp"
echo "Save path: $save_path"


ckpt_dir="$root_dir/runs/$exp/ckpt_model"

# Merge the deepspeed weights to Pytorch Model
echo "<<<<<<<----->>>>>>> Preparing weights for $exp"
cd $ckpt_dir && python zero_to_fp32.py . ../pytorch_model.bin && cd $root_dir

# # # Convert the Pytorch Model to Huggingface Model
echo "<<<<<<<----->>>>>>> Converting weights for $exp to HF"
python $root_dir/merge_lora_weights_and_save_hf_model.py --weight="$root_dir/runs/$exp/pytorch_model.bin" --save_path="$root_dir/trained_models/$save_path"