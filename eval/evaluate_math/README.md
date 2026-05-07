# Math Evaluation

Evaluation scripts for mathematical reasoning benchmarks used in the CGD paper.

## Supported Benchmarks

| Benchmark | Description |
|-----------|-------------|
| MATH-500 | 500 challenging math problems (Minerva subset) |
| Minerva-Math | Quantitative reasoning problems |
| GSM8K | Grade school math word problems |
| OlympiadBench | Competition-level mathematics |
| AMC23 | American Mathematics Competition 2023 |
| AIME24 / AIME25 | American Invitational Math Examination |
| TheoremQA | Theorem-based question answering |

## Requirements

```bash
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt
pip install vllm --no-build-isolation
```

## Usage

### Evaluating LLaMA Models (Primary)

```bash
# LLaMA3.1-8B-Instruct (CGD-trained)
export CUDA_VISIBLE_DEVICES="0"
PROMPT_TYPE="llama3-cot"
MODEL_NAME_OR_PATH="/path/to/your/cgd-trained-model"
bash scripts/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
```

### Evaluating Qwen Models

```bash
# Qwen2.5-Math-7B-Instruct
export CUDA_VISIBLE_DEVICES="0"
PROMPT_TYPE="qwen25-math-cot"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
bash scripts/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH

# Qwen2.5-Math-72B-Instruct (multi-GPU)
export CUDA_VISIBLE_DEVICES="0,1,2,3"
MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-72B-Instruct"
bash scripts/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH
```

### Evaluating on Specific Datasets

```bash
export CUDA_VISIBLE_DEVICES="0"
python evaluate.py \
  --model_name_or_path /path/to/model \
  --dataset math-500 \
  --prompt_type llama3-cot
```

Available `--dataset` options: `math-500`, `minerva-math`, `gsm8k`, `olympiadbench`, `amc23`, `aime24`, `theoremqa`

## Output

Results are saved as JSON files with per-problem accuracy and aggregate statistics. Evaluation uses exact match accuracy with greedy decoding (temperature=0) by default.

## Acknowledgement

The codebase is adapted from [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness).
