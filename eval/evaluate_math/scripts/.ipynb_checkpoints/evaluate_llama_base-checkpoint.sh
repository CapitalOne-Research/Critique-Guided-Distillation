set -ex

# PROMPT_TYPE="direct"
PROMPT_TYPE="cot"
MODEL_NAME_OR_PATH=$1
OUTPUT_DIR=$2
SUMMARY_PATH=$3
SPLIT="test"
NUM_TEST_SAMPLE=-1

cd ..
mkdir -p $OUTPUT_DIR

DATA_NAME="math,minerva_math,gsm8k,olympiadbench,amc23,aime24,theoremqa"
NUM_SHOTS_ARG="math:4,minerva_math:4,gsm8k:4,olympiadbench:0,amc23:0,aime24:0,theoremqa:5"
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
    --num_shots ${NUM_SHOTS_ARG} \
    # --overwrite \
