#!/bin/bash
set -e

# Usage:
#   ./run_eval.sh model_path1 [model_path2 ...]
# A model_path can be:
#   - A checkpoint file (in which case its parent folder will be used), or
#   - A major directory containing subdirectories with checkpoints.
# The script will recursively search for the first directory in each branch that
# contains a *.safetensors file and use that directory as the model_path.

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 model_path1 [model_path2 ...]"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# If a single argument is provided and it contains commas, split it.
if [ "$#" -eq 1 ] && [[ "$1" == *","* ]]; then
  IFS=',' read -r -a provided_paths <<< "$1"
else
  provided_paths=("$@")
fi

# Recursive function to find the top-level directory that contains a *.safetensors file.
# It stops recursing into a branch as soon as it finds a directory with at least one safetensors file.
find_checkpoint_dirs() {
  local dir="$1"
  # If the directory directly contains any .safetensors file, print it and do not descend.
  if compgen -G "$dir"/*.safetensors > /dev/null; then
    echo "${dir%/}"
  else
    # Otherwise, look into each subdirectory.
    for sub in "$dir"/*/; do
      [ -d "$sub" ] && find_checkpoint_dirs "$sub"
    done
  fi
}

# Array to store final model directories for evaluation.
declare -a eval_dirs=()

# Process each provided path.
for path in "${provided_paths[@]}"; do
  if [ -d "$path" ]; then
    echo "Searching for checkpoint directories in: $path"
    while IFS= read -r cp_dir; do
      # Add the directory if it exists.
      if [ -n "$cp_dir" ]; then
        eval_dirs+=("$cp_dir")
      fi
    done < <(find_checkpoint_dirs "$path")
  else
    # If a file is provided, use its parent directory.
    parent_dir=$(dirname "$path")
    eval_dirs+=("${parent_dir%/}")
  fi
done

# Save the original working directory.
ORIG_DIR=$(pwd)

# Loop over each model directory found.
for model_dir in "${eval_dirs[@]}"; do
  echo "------------------------------------------"
  echo "Evaluating model directory: ${model_dir}"

  # Determine the output suffix:
  # If the model directory contains "llama_factory", remove everything up to and including it.
  if [[ "${model_dir}" == *"llama_factory"* ]]; then
    relative_path="${model_dir#*llama_factory/}"
    # Remove trailing slash if any.
    relative_path="${relative_path%/}"
    # Replace forward slashes with underscores.
    output_suffix=$(echo "${relative_path}" | sed 's#/#_#g')
  else
    # Otherwise, use the basename of the directory.
    output_suffix=$(basename "${model_dir}")
  fi

  # Construct the output directory and summary file paths.
  output_dir="../eval_output_cotnshot_${output_suffix}"
  summary_path="../eval_summary_cotnshot_${output_suffix}.txt"

  echo "Output directory: ${output_dir}"
  echo "Summary path: ${summary_path}"

  # Run the evaluations

  # 1. Evaluate on math dataset
  cd ${ORIG_DIR}/../evaluate_math/scripts
  bash evaluate_llama_base.sh ${model_dir} ${output_dir} ${summary_path}

  # 2. Evaluate on GPQA (with extra parameter 5)
  cd ${ORIG_DIR}/../evaluate_gpqa/scripts
  bash evaluate_gpqa.sh "${model_dir}" "${output_dir}" "${summary_path}" 5

  # 3. Evaluate on MMLU-Pro (with extra parameter 5)
  cd ${ORIG_DIR}/../evaluate_mmlu-pro
  bash mmlu-pro-eval.sh "${model_dir}" "${output_dir}" "${summary_path}" 5

  # Return to the original directory.
  cd ${ORIG_DIR}
done
