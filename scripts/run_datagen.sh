#!/bin/bash

root_dir="InteractVLM"
env_root="./"

# Check if the user provided a number as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <number from 0 to n>"
  exit 1
fi

# Assign the argument to a variable
number=$1

##################################### PIAD OBJECT MASK ########################################

# output_dir="rendered_points_heatmap_AP1K0_1104_color"
output_dir="rendered_points_heatmap_1102_color"
data_type="unseen"
split="Test"
# Run the Python program with the selected parameters
${env_root}envs/sdfusion/bin/python ${root_dir}/preprocess_data/generate_piad_obj_heatmap.py \
                                                                          --split "$split" \
                                                                          --run_id "$number" \
                                                                          --data_type "$data_type" \
                                                                          --output_dir "$output_dir"

##################################### LEMON OBJECT MASK ########################################

# Define the possible combinations

view_types=("views_2" "views_2" "views_4" "views_4" "views_6" "views_6" "views_8" "views_8" "views_10" "views_10")
splits=("train" "val" "train" "val" "train" "val" "train" "val" "train" "val")

# Get the specific number from the command line argument
if [ -z "$1" ]; then
    echo "Please provide a number to select the experiment."
    exit 1
fi

i=$1

if [ $i -ge ${#splits[@]} ] || [ $i -lt 0 ]; then
    echo "Invalid number. Please select a number between 0 and $((${#splits[@]} - 1))."
    exit 1
fi

split=${splits[$i]}
view_type=${view_types[$i]}

output_dir="rendered_points_heatmap_1030_color"
# Run the Python program with the selected parameters
${env_root}envs/sdfusion/bin/python ${root_dir}/preprocess_data/generate_lemon_obj_heatmap.py \
                                                                          --split "$split" \
                                                                          --view_type "$view_type" \
                                                                          --output_dir "$output_dir"

#################################### LEMON HUMAN MASK ########################################
# Define the possible combinations
case $number in
  0)
    split="train"
    ;;
  1)
    split="val"
    ;;
  *)
    echo "Invalid number! Please provide a number between 0 and 1."
    exit 1
    ;;
esac

view_type="views4"
output_dir="hcontact_vitruvian"

${env_root}envs/sdfusion/bin/python ${root_dir}/preprocess_data/generate_lemon_human_mask.py \
                                                                          --split "$split" \
                                                                          --output_dir "$output_dir" \
                                                                          --view_type "$view_type"

#################################### DAMON HUMAN MASK ########################################

case $number in
  0)
    split="train"
    mask_type="objectwise"
    ;;
  1)
    split="train"
    mask_type="all"
    ;;
  2)
    split="test"
    mask_type="objectwise"
    ;;
  3)
    split="test"
    mask_type="all"
    ;;
  *)
    echo "Invalid number! Please provide a number between 0 and 1."
    exit 1
    ;;
esac

view_type="views4"
output_dir="hcontact_vitruvian"

${env_root}envs/sdfusion/bin/python ${root_dir}/preprocess_data/generate_damon_human_mask.py \
                                                                          --split "$split" \
                                                                          --mask_type "$mask_type" \
                                                                          --output_dir "$output_dir" \
                                                                          --view_type "$view_type"