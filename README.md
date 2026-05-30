# 🚀 SOTA Multi-GPU GRPO Reasoning Project (T4x2 Optimized)

This project contains a highly optimized, config-driven reinforcement learning post-training pipeline to align `Qwen/Qwen2.5-Math-7B-Base` on mathematical reasoning tasks using **Group Relative Policy Optimization (GRPO)**. It is built specifically for **Kaggle T4×2 (or dual-GPU) environments**, utilizing both GPUs via Distributed Data Parallel (DDP) mode.

---

## 🎯 Key Capabilities Developed

1. **System 2 Chain-of-Thought (CoT)**: Aligns the model to generate intermediate reasoning paths inside `<think>...</think>` tags to self-correct and verify math steps before returning a final answer.
2. **Structured Formatting Compliance**: Enforces LaTeX `\boxed{}` outputs for final numerical answers.
3. **Dataset Agnosticism**: Evaluates mathematical solutions dynamically across different schemas (e.g. Olympiad, high school, or basic math word problems).

---

## 📂 Project Structure

```text
multi-gpu-grpo-reasoning/
├── config.yaml              # Central configuration (Single Source of Truth)
├── data_prep.py             # Dataset-agnostic streaming and preprocessing
├── grpo_train.py            # DDP Multi-GPU GRPO Trainer with custom rewards
├── eval_export.py           # Evaluation traces, GPU-based merging, and GGUF export
├── requirements.txt         # Dependencies (no strict version pins for Kaggle)
├── run_all.sh               # Shell script orchestrator (executable)
└── README.md                # This documentation file
```

---

## ⚙️ How It Works (Under the Hood)

### 1. Dynamic Dataset Mapping (`data_prep.py`)
Rather than forcing a specific schema, the data preparation script streams the target dataset (default: `AI-MO/NuminaMath-CoT`) and dynamically maps prompt and answer fields. It automatically detects fields named `problem`/`question`/`prompt` and `solution`/`answer`/`completion`/`reasoning`. 

### 2. Multi-GPU Device Allocation (`grpo_train.py`)
To run a 7B model across multiple GPUs without cross-process VRAM leaks, the script bypasses `device_map="auto"` (which shards a single model copy across multiple cards) and instead sets up local device ranks:
```python
device_map = {"": local_rank}
```
This forces each rank to train its own full model copy on its designated GPU, executing DDP correctly.

### 3. Multi-Format Verifiable Rewards (`grpo_train.py`)
The correctness reward function parses both the model output and the ground truth for mathematical values. It extracts answers wrapped in `\boxed{}`, GSM8K-style `#### <val>`, or raw numbers, comparing them directly to award scores.

### 4. Low System RAM Adapter Merging (`eval_export.py`)
Merging a 7B model in 16-bit precision takes ~14GB of memory. Doing this in system RAM on Kaggle (limit: 16GB) frequently causes Out-Of-Memory (OOM) crashes. `eval_export.py` loads the base model directly in GPU memory to perform the merge via PEFT:
```python
model = model.merge_and_unload()
```
This stays safely within the 16GB VRAM of a single GPU, keeping system RAM usage near zero.

---

## 🚀 Step-by-Step Run Guide

### 1. Configure the Run (`config.yaml`)
You can adjust the hyperparameters, model target, or Hugging Face dataset in `config.yaml`:
```yaml
data:
  dataset: "AI-MO/NuminaMath-CoT"
  split: "train"
  max_examples: 1000

training:
  grpo:
    epochs: 1
    lr: 1e-6
    num_generations: 4
```

### 2. Run the Pipeline
Upload this directory to Kaggle, ensure internet access is enabled, and run the pipeline driver:
```bash
chmod +x run_all.sh
./run_all.sh
```

The script will:
* Detect the number of available GPUs on your system.
* Install dependencies and cache data safely under `/tmp/`.
* Preprocess and cache the mathematical datasets.
* Launch DDP training using `accelerate launch --multi_gpu --num_processes=$NUM_GPUS grpo_train.py`.
* Run evaluation checks on mathematical queries.
* Merge the adapter with the base model and export the model to `outputs/math_reasoning_qwen2.5_7b.gguf`.
