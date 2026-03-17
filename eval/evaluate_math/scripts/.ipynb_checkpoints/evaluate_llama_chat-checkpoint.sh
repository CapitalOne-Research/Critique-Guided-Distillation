set -ex

PROMPT_TYPE="llama3-math-cot"
MODEL_NAME_OR_PATH=$1
OUTPUT_DIR=$2
SUMMARY_PATH=$3
SPLIT="test"
NUM_TEST_SAMPLE=-1

cd ..
mkdir -p $OUTPUT_DIR

# DATA_NAME="math,minerva_math,gsm8k,olympiadbench,amc23,aime24,theoremqa"
DATA_NAME="math-500,minerva_math,gsm8k,olympiadbench,amc23,aime24,theoremqa"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --summary_path ${SUMMARY_PATH} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --max_tokens_per_call 8192 \
    # --apply_chat_template \
    # --overwrite \