#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# model_dir="/path/to/models"
# model_dir="/llm-experiments-no-cache/beco/model_runs/llama_factory/ultra_and_magpie31/gradAcc4_2epochs"
# model_dir="/llm-experiments-no-cache/beco/model_runs/llama_factory/cft/packing/F1_CPT_V4/llama3p3_70b_OMI2/gradAcc1_1epochs"

model_dir="/llm-experiments-no-cache/beco/model_runs/llama_factory/csd/packing/llama3p1_8B_Inst/wi_ch1_225k/64gbs_gradAcc4_7epochs_1e6/"
summary_path="../../Validation/packing/csd/llama3p1_8B_Inst/full/wi_ch1_225k/64gbs_gradAcc4_7epochs_1e6/validation_summary.txt"

models_dir_name=$(basename "$model_dir")
summary_parent_dir=$(dirname "$summary_path")

min_ckp=1
max_ckp=50000

for checkpoint_dir in ${model_dir}/checkpoint-*; do
    if [ -d "$checkpoint_dir" ]; then

        checkpoint_num=$(basename "$checkpoint_dir" | cut -d'-' -f2)
        if [ "$checkpoint_num" -ge "$min_ckp" ] && [ "$checkpoint_num" -lt "$max_ckp" ]; then

            output_dir="${summary_parent_dir}/${models_dir_name}-checkpoint-${checkpoint_num}/"

            echo "Processing checkpoint-${checkpoint_num}"
            echo "Model path: ${checkpoint_dir}"
            echo "Output dir: ${output_dir}"

            bash validate_single.sh "$checkpoint_dir" "$output_dir" "$summary_path"
        else
            echo "Skipping checkpoint-${checkpoint_num} as it's >= ${max_ckp}"
        fi
    fi
done
