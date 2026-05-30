import os
import sys
import gc
import re
import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from omegaconf import OmegaConf

# 1. Answer Extraction helper functions
def extract_boxed_answer(completion: str) -> str:
    # Match the content inside \boxed{...}
    # Handling nested brackets or common spacing
    matches = re.findall(r"\\boxed\s*\{\s*([^{}]+?)\s*\}", completion)
    if matches:
        # Get the last match, which represents the final answer
        ans = matches[-1].strip()
        # Clean currency symbols, commas, or extra units
        ans = ans.replace("$", "").replace(",", "").replace("%", "").strip()
        # Extract just the numerical part if mixed with text (e.g. "120 apples" -> "120")
        num_match = re.search(r"(-?\d+(?:\.\d+)?)", ans)
        if num_match:
            return num_match.group(1)
        return ans
    return ""

def extract_gt_answer(answer_str: str) -> str:
    # 1. Look for \boxed{...} in ground truth (used in NuminaMath-CoT, hendrycks_math, PURE)
    matches = re.findall(r"\\boxed\s*\{\s*([^{}]+?)\s*\}", answer_str)
    if matches:
        ans = matches[-1].strip()
        ans = ans.replace("$", "").replace(",", "").replace("%", "").strip()
        num_match = re.search(r"(-?\d+(?:\.\d+)?)", ans)
        if num_match:
            return num_match.group(1)
        return ans
        
    # 2. Look for GSM8K-style #### <number>
    match = re.search(r"####\s*(-?\d+)", answer_str)
    if match:
        return match.group(1).strip()
        
    # 3. Fallback: if it's a short string, try to parse a number directly
    cleaned = answer_str.strip().replace("$", "").replace(",", "").replace("%", "").strip()
    num_match = re.search(r"(-?\d+(?:\.\d+)?)", cleaned)
    if num_match:
        return num_match.group(1)
        
    return cleaned

# 2. Correctness Reward Function
def correctness_reward_func(prompts, completions, answer, **kwargs):
    rewards = []
    for completion, gt in zip(completions, answer):
        gt_val = extract_gt_answer(gt)
        model_val = extract_boxed_answer(completion)
        
        # 💡 DEBUG LOGGING: Print active generations so the console doesn't look frozen!
        if os.environ.get("LOCAL_RANK", "0") == "0":
            preview = completion.replace('\n', ' ')
            print(f"\n[GPU 0 Active Rollout] 🤖 {preview[:120]}... \n[Target: {gt_val} | Extracted: {model_val}]")
            
        if gt_val and model_val == gt_val:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

# 3. Format Reward Function
def format_reward_func(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        # Check if model formatted thinking process inside <think>...</think>
        has_think_start = "<think>" in completion
        has_think_end = "</think>" in completion
        has_boxed = "\\boxed" in completion
        
        score = 0.0
        if has_think_start:
            score += 0.25
        if has_think_end:
            score += 0.25
        if has_boxed:
            score += 0.50
            
        rewards.append(score)
    return rewards

def main():
    # ⚠️ Safeguard to warn users if they run the script directly instead of via accelerate
    if torch.cuda.device_count() > 1 and "LOCAL_RANK" not in os.environ:
        print("\n⚠️ WARNING: Multiple GPUs detected but DDP is not initialized!")
        print("💡 To utilize all GPUs, you MUST launch this script using accelerate:")
        print("   accelerate launch --multi_gpu --num_processes 2 grpo_train.py")
        print("   (Or simply execute the ./run_all.sh script)\n")
        
    # NATIVE FAILSAFE: Force PyTorch to default to float16 to prevent ANY bfloat16 leakage on T4 GPUs
    torch.set_default_dtype(torch.float16)

    # Get local rank for DDP device mapping (avoids Accelerator conflict with Trainer)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(config_path):
        if local_rank == 0:
            print("Error: config.yaml not found!")
        sys.exit(1)
        
    cfg = OmegaConf.load(config_path)
    
    # Setup paths and environment
    base_path = "."
    if "KAGGLE_WORKING_DIR" in os.environ:
        base_path = os.environ["KAGGLE_WORKING_DIR"]
    elif "/content" in os.getcwd():
        base_path = "/content"
        
    output_dir = os.path.join(base_path, cfg.paths.output)
    cfg.paths.output = output_dir
    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        
    cache_dir = os.environ.get("CACHE_DIR", "/tmp/transformers_cache")
    
    # Let Hugging Face Trainer handle W&B automatically across DDP
    report_to = "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    if report_to == "wandb":
        os.environ["WANDB_PROJECT"] = cfg.paths.wandb_project

    # BitsAndBytesConfig for 4-bit QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    if local_rank == 0:
        print(f"Loading tokenizer for base: {cfg.model.base}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base, cache_dir=cache_dir)
    
    # Configure pad token if it is missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if local_rank == 0:
        print(f"Loading base model {cfg.model.base} in 4-bit on GPU {local_rank}...")
        
    # PRE-LOAD CONFIG: Qwen natively uses bfloat16. We MUST override it before instantiation
    # so internal components like RoPE embeddings don't initialize as bfloat16 and poison gradients.
    model_config = AutoConfig.from_pretrained(cfg.model.base, cache_dir=cache_dir)
    model_config.torch_dtype = torch.float16

    # Load model with correct device mapping to prevent cross-GPU bloat in DDP
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base,
        config=model_config,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    )

    # Prepare model for k-bit training and configure gradient checkpointing
    model = prepare_model_for_kbit_training(
        model, 
        use_gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Define LoRA Configuration
    peft_config = LoraConfig(
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.lora_alpha,
        target_modules=list(cfg.model.lora.target_modules),
        lora_dropout=cfg.model.lora.dropout,
        bias=cfg.model.lora.bias,
        task_type=cfg.model.lora.task_type
    )
    
    # Wrap model with LoRA
    model = get_peft_model(model, peft_config)
    

    # Load prepared math dataset
    grpo_cached_path = os.path.join(cfg.data.cache_dir, "grpo_cached")
    if local_rank == 0:
        print(f"Loading cached math dataset from {grpo_cached_path}...")
    if not os.path.exists(grpo_cached_path):
        if local_rank == 0:
            print("Error: Cached dataset does not exist. Run data_prep.py first.")
        sys.exit(1)
        
    grpo_ds = datasets.load_from_disk(grpo_cached_path)

    # Set up GRPO Config
    grpo_config = GRPOConfig(
        output_dir=os.path.join(cfg.paths.output, "grpo"),
        per_device_train_batch_size=cfg.hardware.batch_size,
        gradient_accumulation_steps=cfg.hardware.grad_accum,
        optim=cfg.hardware.optim,
        learning_rate=cfg.training.grpo.lr,
        num_train_epochs=cfg.training.grpo.epochs,
        num_generations=cfg.training.grpo.num_generations,
        warmup_steps=cfg.training.grpo.warmup_steps,
        lr_scheduler_type="cosine",
        fp16=cfg.hardware.fp16,
        bf16=cfg.hardware.bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.training.grpo.logging_steps,
        save_steps=cfg.training.grpo.save_steps,
        save_total_limit=2,
        report_to=report_to,
        ddp_find_unused_parameters=False,  # Essential optimization flag for DDP
        logging_first_step=True
    )

    if local_rank == 0:
        print("🚀 Instantiating GRPOTrainer under DDP...")
        
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=grpo_ds,
        args=grpo_config,
        reward_funcs=[correctness_reward_func, format_reward_func]
    )

    # Train model
    trainer.train()
    
    # Save training adapters
    if local_rank == 0:
        print(f"💾 Saving trained GRPO adapters to {os.path.join(cfg.paths.output, 'grpo')}...")
        trainer.save_model()
        tokenizer.save_pretrained(os.path.join(cfg.paths.output, "grpo"))

    # Cleanup memory
    del trainer
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    if local_rank == 0:
        print("✅ Multi-GPU GRPO math alignment completed successfully!")

if __name__ == "__main__":
    main()
