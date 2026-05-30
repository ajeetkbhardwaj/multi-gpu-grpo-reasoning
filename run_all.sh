#!/usr/bin/env bash
set -e

# SOTA Multi-GPU GRPO Reasoning Project Runner
# Auto-detects GPUs, sets up paths, prepares dataset, and executes training + evaluation.

export CACHE_DIR="/tmp/transformers_cache"
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache"
export WANDB_DISABLED="false"

echo "📂 Creating cache directories..."
mkdir -p "$CACHE_DIR"
mkdir -p "$HF_DATASETS_CACHE"

echo "📦 Installing project dependencies..."
pip install -r requirements.txt

# Sanity check for GPU presence
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ nvidia-smi not found. Running on CPU is not recommended for GRPO, but executing scripts anyway..."
    NUM_GPUS=0
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

echo "📊 Detected GPUs: $NUM_GPUS"

echo "📥 Step 1: Pre-curating and formatting math dataset from HuggingFace..."
python data_prep.py

echo "🚀 Step 2: Starting GRPO Policy Training..."
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "⚡ Launching Distributed Data Parallel (DDP) across $NUM_GPUS GPUs..."
    # Launch training using accelerate in multi-GPU mode
    accelerator_args=(
        "--multi_gpu"
        "--num_machines" "1"
        "--num_processes" "$NUM_GPUS"
        "--dynamically_quantized_training" "False"
    )
    accelerate launch "${accelerator_args[@]}" grpo_train.py
else
    echo "⚡ Launching on a single GPU..."
    python grpo_train.py
fi

echo "🔋 Step 3: Evaluating model traces and exporting to GGUF format..."
python eval_export.py

echo "🏁 All steps completed successfully!"
