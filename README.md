# Critique-Guided Distillation (CGD)
Critique-Guided Distillation for Robust Reasoning via Refinement

This repo contains the code for [Critique-Guided Distillation for Robust Reasoning via Refinement](https://arxiv.org/abs/2505.11628). In this paper, we introduce Critique-Guided Distillation (CGD) - a novel multi-step fine-tuning paradigm in which a student model learns to **transform** its own initial outputs into high-quality refinements—rather than merely generating critiques!  

<a target="_blank" href="https://github.com/CapitalOne-Research/Critique-Guided-Distillation">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-black?style=flat&logo=github"></a>
<a target="_blank" href="https://arxiv.org/abs/2505.11628">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-green?style=flat&logo=arxiv"></a>
<br>

## Highlights
CGD consistently outperforms standard distillation and Critique Fine-Tuning (CFT) across mathematical and general reasoning benchmarks!

![CGD Results Visualization](res/accuracy_combined.png)

---

## News
- **[2026/03/18]** ⚡️ The paper, code, data, and model for Critique-Guided Distillation (CGD) are all available online.



## 💡 How It Works

CGD creates a powerful training signal by teaching the student model to refine its own work. The process consists of two main phases: data generation and fine-tuning.

1.  **Probe:** The weaker **student model** generates an initial answer to a prompt.
2.  **Critique:** A stronger **teacher model** critiques the student's answer, identifying errors and suggesting improvements.
3.  **Refine:** The **teacher model** writes a gold-standard, refined answer based on the original prompt, the student's attempt, and its own critique.
4.  **Fine-Tune:** The student model is then fine-tuned on a dataset where the `input` is the original prompt and the teacher's critique, and the `output` is the refined answer. This teaches the student *how to process criticism and improve*.

---

## 🚀 Getting Started

Follow these steps to generate the CGD dataset and perform CGD training on the student model.

### 🛠️ Step 1: Setup

First, clone the repository and install the necessary dependencies. It's recommended to use a virtual environment.

```bash
# Clone the repository
git clone [https://github.com/CapitalOne-Research/Critique-Guided-Distillation](https://github.com/CapitalOne-Research/Critique-Guided-Distillation)
cd critique-guided-distillation

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Run the Data Generation Pipeline
The core of this project is the data generation script, which automates the Probe, Critique, and Refine steps.

#### A. Configure the Pipeline
Before running, you must open run_pipeline.sh and configure the following variables at the top of the file:

```bash
# --- Configuration ---

# Model used to generate initial responses (your student model)
STUDENT_MODEL="/path/to/your/student_model_70b" # <--- ⚠️ UPDATE THIS PATH

# Stronger model used to critique and refine responses (your teacher model)
TEACHER_MODEL="/path/to/your/teacher_model_70b" # <--- ⚠️ UPDATE THIS PATH

# Name of the initial dataset configured in pipeline.py
INITIAL_DATASET_NAME="webinstruct_sub"

# Directory to save all intermediate and final files
OUTPUT_DIR="./../data"

# vLLM tensor parallel size (number of GPUs)
TP_SIZE=8

# --- ⚙️ Sampling and Performance ---

# Set the number of samples you want to process from the initial dataset.
NUM_SAMPLES=25000
```

#### B. Execute the Script
Once configured, run the script from your terminal. It will execute all three data generation steps sequentially and print timing information for each.

```bash
bash run_pipeline.sh
```


The script will create an output directory with the following structure:

- data/01_probed_responses.json: Initial answers from the student model.

- data/02_critiqued_responses.json: Critiques from the teacher model.

- data/03_refined_dataset.json: The final, SFT-ready dataset.

The final dataset is automatically formatted for fine-tuning, with each entry containing an instruction, input, and output key.

### Step 3: Fine-Tuning with LLaMA Factory
The generated dataset can now be used to fine-tune your student model. If you haven't already, install LLaMA Factory.

```bash
git clone [https://github.com/hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory)
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

#### A. Create a Dataset Config File
In the LLaMA-Factory/data directory, create a dataset_info.json file entry for the CGD data.

```json5
{
  "cgd_data": {
    "file_name": "/path/to/your/critique-guided-distillation/data/03_refined_dataset.json"
  }
}
```

#### B. Start Fine-Tuning
You can now launch the fine-tuning job using LLaMA Factory.

---

## 📊 Results

### WebInstruct Fine-Tuning on LLaMA 3.1-8B

CGD consistently achieves better results than standard Supervised Fine-Tuning (SFT), Distilled SFT, and Critique Fine-Tuning (CFT) across both math and general reasoning tasks.

| Method                          | MATH500 | Minerva-Math | GSM8K | OlympiadBench | AMC23 | **Avg. (G1)** | TheoremQA | GPQA | MMLU-PRO | **Avg. (G2)** |
|---------------------------------|---------|--------------|-------|---------------|-------|---------------|-----------|------|----------|---------------|
| LLaMA3.1-8B Instruct            | 50.6    | 33.5         | 85.3  | 14.5          | 22.5  | 41.3          | 27.6      | 30.8 | 31.2     | 29.9          |
| + SFT                           | 41.2    | 24.6         | 80.7  | 10.8          | 20.0  | 35.5          | 22.1      | 33.3 | 39.3     | 31.6          |
| + Distilled SFT                 | 53.4    | 32.7         | 85.3  | 19.6          | 27.5  | 43.7          | 28.9      | 31.8 | 35.1     | 31.9          |
| + CFT                           | 51.8    | 32.7         | 84.8  | 15.7          | 22.5  | 41.5          | 28.2      | 34.3 | 34.2     | 32.4          |
| + CFT* (GPT-4o)                 | 54.8    | 33.1         | **86.2** | 18.2       | 25.0  | 43.5          | **35.0** | 30.3 | **40.8** | 36.4          |
| + CGD (ours)                    | **54.2**| **33.6** | 85.7  | **23.7** | **37.5**| **46.9** | 34.0      | **35.9**| 40.3  | **36.7** |
| **Δ (CGD − CFT)** | +2.4    | +0.9         | +0.9  | +8.0          | +15.0 | +5.4          | +5.8      | +1.6 | +6.1     | +4.3          |

*Note: CFT uses the same 100K samples as the other experiments. CFT\* (GPT-4o) uses only 50K samples from WebInstructCFT. CGD uses only 100K questions from the WebInstruct dataset.*

### Cross-Family Validation on Qwen2.5-Math-7B

CGD demonstrates strong cross-family effectiveness. Even when using a weaker open-source teacher (S1.1-32B), CGD maintains competitive performance and significantly outperforms CFT.

| Method | Teacher Model | MATH500 | Minerva-Math | OlympiadBench | AMC23 | AIME24 | **Avg.** |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B (Base) | - | 55.4 | 13.6 | 19.9 | 40.0 | 10.0 | 27.8 |
| CFT | GPT-4o | 79.2 | 45.2 | 40.7 | 62.5 | 16.7 | 48.9 |
| **CGD (Ours)** | **Claude Sonnet 3.7** | 79.4 | 44.1 | 41.2 | **67.5** | **20.0** | **50.4** |
| **CGD (Ours)** | **S1.1-32B** | **79.6** | **48.5** | **41.3** | 62.5 | 13.3 | 49.0 |

### Out-of-Distribution Generalization

Unlike CFT, which severely degrades general capabilities (e.g., causing a -21.3% drop on IFEval), CGD safely preserves and even improves out-of-distribution performance, such as code generation (HumanEval), despite being trained on data without code.

| Method | IFEval | MUSR | TruthfulQA | BBH | HumanEval |
|---|---|---|---|---|---|
| LLaMA3.1-8B Instruct | 76.9 | 37.8 | 54.0 | **48.3** | 59.7 |
| + SFT | 76.6 | 36.9 | 52.0 | 48.0 | 57.8 |
| + Distilled SFT | **77.5** | 39.0 | 53.9 | 47.0 | 58.7 |
| + CFT w/ GPT-4o | 55.6 | 35.0 | 53.5 | 44.2 | 60.3 |
| **+ CGD (ours)** | 76.1 | **39.3** | **54.5** | 47.1 | **64.6** |

**Legend:**

- Bold = best

- Italic = second-best

- CFT uses the same 100K samples as the other experiments

- CFT* (GPT-4o) uses only 50K samples from WebInstructCFT

- CGD uses only 100K questions from WebInstruct dataset