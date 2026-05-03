# CGD Data Generation Pipeline

This directory contains the core data generation pipeline for Critique-Guided Distillation.

## Overview

The pipeline generates training data in three sequential steps using a student-teacher setup:

| Step | Script Call | Description | Output |
|------|-----------|-------------|--------|
| 1. **Probe** | `cgd_pipeline.py --step probe` | Student model generates initial answers | `01_probed_responses.json` |
| 2. **Critique** | `cgd_pipeline.py --step critique` | Teacher model critiques student answers | `02_critiqued_responses.json` |
| 3. **Refine** | `cgd_pipeline.py --step refine` | Teacher generates refined answers given critique | `03_refined_dataset.json` |

The final output (`03_refined_dataset.json`) is formatted for SFT and can be directly used with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

## Prerequisites

- **vLLM** installed (`pip install vllm`)
- GPU(s) with enough memory for the teacher model (e.g., 8xA100 for 70B models)
- Student and teacher model weights downloaded locally

## Usage

### Quick Start (end-to-end)

Edit `run_cgd_pipeline.sh` to set your model paths, then:

```bash
bash run_cgd_pipeline.sh
```

### Configuration

Key variables in `run_cgd_pipeline.sh`:

```bash
STUDENT_MODEL="/path/to/Llama-3.1-8B-Instruct"    # Student model path
TEACHER_MODEL="/path/to/Llama-3.3-70B-Instruct"    # Teacher model path
INITIAL_DATASET_NAME="webinstruct_sub"              # Dataset name (configured in cgd_pipeline.py)
TP_SIZE=8                                            # Tensor parallel size (number of GPUs)
NUM_SAMPLES=100000                                   # Number of samples to process ("" for all)
```

### Running Steps Individually

You can also run each step separately:

```bash
# Step 1: Generate student answers
python cgd_pipeline.py \
  --step probe \
  --model_path /path/to/student_model \
  --dataset_name webinstruct_sub \
  --output_file data/01_probed_responses.json \
  --tensor_parallel_size 8 \
  --num_samples 100000

# Step 2: Generate teacher critiques
python cgd_pipeline.py \
  --step critique \
  --model_path /path/to/teacher_model \
  --input_file data/01_probed_responses.json \
  --output_file data/02_critiqued_responses.json \
  --tensor_parallel_size 8

# Step 3: Generate refined answers
python cgd_pipeline.py \
  --step refine \
  --model_path /path/to/teacher_model \
  --input_file data/02_critiqued_responses.json \
  --output_file data/03_refined_dataset.json \
  --tensor_parallel_size 8
```

## Files

| File | Description |
|------|-------------|
| `cgd_pipeline.py` | Main pipeline script with probe/critique/refine steps |
| `run_cgd_pipeline.sh` | End-to-end execution wrapper |
| `filter_crit_ratios.py` | Utility for filtering critique quality based on syntactic/semantic similarity |

## Supported Datasets

The pipeline supports multiple datasets configured in `cgd_pipeline.py`:
- `webinstruct_sub` / `webinstruct_filter` — WebInstruct (default)
- `openmathinstruct2` — OpenMathInstruct-2
- `metamathqa` — MetaMathQA

## Output Format

The final dataset (`03_refined_dataset.json`) contains entries formatted for SFT:

```json
{
  "instruction": "<original prompt>",
  "input": "<student answer + teacher critique>",
  "output": "<teacher's refined answer>"
}
```
