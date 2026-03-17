#!/bin/bash
set -e

export VLLM_WORKER_MULTIPROC_METHOD=spawn

total_start_time=$(date +%s)

# --- 🚀 Configuration ---

# Student model (probe)
STUDENT_MODEL="/llm-experiments-no-cache/model_zoo/Llama-3.1-8B"

# Teacher model (critique/refine) — GPT-OSS
# TEACHER_MODEL="/model-zoo/Huggingface/openai/gpt-oss-120b"
# TEACHER_MODEL="/model_zoo/Meta-Llama-3.1-70B-Instruct"
# TEACHER_MODEL="/model_zoo/simplescaling_s1.1-32B"
TEACHER_MODEL="/llm-experiments-no-cache/model_zoo/Llama-3.3-70B-Instruct"

# Dataset
# INITIAL_DATASET_NAME="webinstruct_sub"
INITIAL_DATASET_NAME="webinstruct_filter"

# Output directory
OUTPUT_DIR="./../data/Llama-3.1-8B/Llama-3.3-70B-Instruct/${INITIAL_DATASET_NAME}"
mkdir -p "$OUTPUT_DIR"

TP_SIZE=8
MAX_STUDENT_LEN=8192
# MAX_STUDENT_LEN=512
MAX_TEACHER_LEN=32768
NUM_SAMPLES=8  # or "" for all
# NUM_SAMPLES=100000  # or "" for all

PROBED_FILE="$OUTPUT_DIR/01_probed_responses.json"
CRITIQUED_FILE="$OUTPUT_DIR/02_critiqued_responses.json"
REFINED_FILE="$OUTPUT_DIR/03_refined_dataset.json"

echo "✅ Outputs -> $OUTPUT_DIR"
echo "Processing ${NUM_SAMPLES:-all} samples with TP_SIZE=$TP_SIZE"

# --- Step 1: Probe (Student) ---
echo -e "\n--- STEP 1: PROBING (Student) ---"
python cgd_pipeline.py \
  --step "probe" \
  --model_path "$STUDENT_MODEL" \
  --dataset_name "$INITIAL_DATASET_NAME" \
  --output_file "$PROBED_FILE" \
  --tensor_parallel_size "$TP_SIZE" \
  --max_model_len "$MAX_STUDENT_LEN" \
  --num_samples "$NUM_SAMPLES"

# --- Step 2: Critique ---
echo -e "\n--- STEP 2: CRITIQUING ---"
python cgd_pipeline.py \
  --step "critique" \
  --model_path "$TEACHER_MODEL" \
  --input_file "$PROBED_FILE" \
  --output_file "$CRITIQUED_FILE" \
  --tensor_parallel_size "$TP_SIZE" \
  --max_model_len "$MAX_TEACHER_LEN"

# --- Step 3: Refine ---
echo -e "\n--- STEP 3: REFINING ---"
python cgd_pipeline.py \
  --step "refine" \
  --model_path "$TEACHER_MODEL" \
  --input_file "$CRITIQUED_FILE" \
  --output_file "$REFINED_FILE" \
  --tensor_parallel_size "$TP_SIZE" \
  --max_model_len "$MAX_TEACHER_LEN"

# --- Final ---
total_end_time=$(date +%s)
total_elapsed=$((total_end_time - total_start_time))
echo -e "\n✅ Pipeline complete! Final refined dataset: ${REFINED_FILE}"
echo "--- ⏱️ Total elapsed: ${total_elapsed} seconds. ---"
