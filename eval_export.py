import os
import sys
import gc
import torch
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from omegaconf import OmegaConf

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n"
    "The assistant first thinks about the reasoning process inside <think> and </think> tags, "
    "and then provides the final answer. The final answer must be enclosed in \\boxed{}, "
    "for example \\boxed{60}."
)

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(config_path):
        print("Error: config.yaml not found!")
        sys.exit(1)
        
    cfg = OmegaConf.load(config_path)
    
    base_path = "."
    if "KAGGLE_WORKING_DIR" in os.environ:
        base_path = os.environ["KAGGLE_WORKING_DIR"]
    elif "/content" in os.getcwd():
        base_path = "/content"
        
    output_dir = os.path.join(base_path, cfg.paths.output)
    grpo_adapter_path = os.path.join(output_dir, "grpo")
    merged_output_path = os.path.join(output_dir, "grpo_merged")
    gguf_output_path = os.path.join(output_dir, cfg.export.gguf_name)
    
    cache_dir = os.environ.get("CACHE_DIR", "/tmp/transformers_cache")
    
    print("🔋 Evaluating GRPO Aligned model...")
    tokenizer = AutoTokenizer.from_pretrained(grpo_adapter_path)
    
    # Pad token check
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load 4-bit config for evaluation
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    print("Loading model in 4-bit for evaluation rollouts...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    )
    
    # Force config to fp16 to ensure hardware-accelerated generation on T4 GPUs
    base_model.config.torch_dtype = torch.float16
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, grpo_adapter_path)
    model.eval()
    
    # Define test questions
    test_questions = [
        "If Mary has 5 apples and John gives her 3 more, then she eats 2, how many apples does she have left?",
        "Solve for x: 3x + 7 = 19."
    ]
    
    for i, q in enumerate(test_questions):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
        # Safely extract only the newly generated tokens using tensor slicing
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"\n📝 Question {i+1}: {q}")
        print(f"🤖 Model response:\n{response}")
        print("-" * 50)
        
    # Free memory
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n🔄 Merging model adapters and exporting to 16-bit...")
    # Load base model in fp16 on CPU (safest for 16GB GPUs to avoid OOM during merge)
    base_model_fp16 = AutoModelForCausalLM.from_pretrained(
        cfg.model.base,
        torch_dtype=torch.float16,
        device_map="cpu",
        cache_dir=cache_dir
    )
    
    model_fp16 = PeftModel.from_pretrained(base_model_fp16, grpo_adapter_path)
    print("Merging adapters with base model...")
    merged_model = model_fp16.merge_and_unload()
    
    print(f"Saving merged model to {merged_output_path}...")
    merged_model.save_pretrained(merged_output_path, safe_serialization=False)
    tokenizer.save_pretrained(merged_output_path)
    
    del merged_model, base_model_fp16, model_fp16
    gc.collect()
    torch.cuda.empty_cache()
    
    # 5. GGUF Conversion using llama-cpp-python
    print(f"🔄 Converting {merged_output_path} to GGUF format...")
    os.makedirs(os.path.dirname(gguf_output_path), exist_ok=True)
    
    try:
        # Attempting conversion using the modern transformers/gguf tool if installed
        import gguf
        print("Please run conversion manually by cloning llama.cpp:")
        print(f"git clone https://github.com/ggerganov/llama.cpp")
        print(f"python llama.cpp/convert_hf_to_gguf.py {merged_output_path} --outfile {gguf_output_path} --outtype q8_0")
        print("For now, skipping automated GGUF conversion to prevent script failure.")
    except Exception as e:
        print(f"⚠️ GGUF conversion encountered an issue: {e}")
        print("Model is saved in merged 16-bit precision format in Outputs directory.")

if __name__ == "__main__":
    main()
