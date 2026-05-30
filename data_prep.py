import os
import sys
import datasets
from transformers import AutoTokenizer
from omegaconf import OmegaConf

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n"
    "The assistant first thinks about the reasoning process inside <think> and </think> tags, "
    "and then provides the final answer. The final answer must be enclosed in \\boxed{}, "
    "for example \\boxed{60}."
)

def main():
    # Load configuration from current directory
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found!")
        sys.exit(1)
        
    cfg = OmegaConf.load(config_path)
    
    # Auto-detect cache directory
    cache_dir = os.environ.get("CACHE_DIR", "/tmp/transformers_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(cfg.data.cache_dir, exist_ok=True)
    
    print(f"Initializing tokenizer for {cfg.model.base}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base, cache_dir=cache_dir)
    
    # In case the tokenizer doesn't have a chat template, define the standard ChatML template
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
        print("💡 Added default ChatML template to tokenizer.")

    # Load dataset with subset handling if present
    print(f"📥 Loading dataset: {cfg.data.dataset}...")
    kwargs = {}
    if "subset" in cfg.data and cfg.data.subset:
        kwargs["name"] = cfg.data.subset
        
    dataset = datasets.load_dataset(cfg.data.dataset, split=cfg.data.split, streaming=True, cache_dir=cache_dir, **kwargs)
    dataset = dataset.take(cfg.data.max_examples)
    
    # Extract columns to remove and detect input/output columns
    first_item = next(iter(dataset))
    remove_cols = list(first_item.keys())
    
    # Detect question column dynamically
    q_col = None
    for col in ["problem", "question", "prompt", "input"]:
        if col in remove_cols:
            q_col = col
            break
            
    # Detect answer/solution column dynamically
    a_col = None
    for col in ["solution", "answer", "completion", "output", "reasoning"]:
        if col in remove_cols:
            a_col = col
            break
            
    if q_col is None or a_col is None:
        raise ValueError(f"Could not identify question/answer columns in dataset. Found fields: {remove_cols}")
        
    print(f"🔎 Detected question column: '{q_col}' | answer column: '{a_col}'")

    # Formatting dataset for GRPO
    def format_grpo(examples):
        prompts = []
        answers = []
        for q, a in zip(examples[q_col], examples[a_col]):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}
            ]
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt_str)
            answers.append(a)
        return {
            "prompt": prompts,
            "answer": answers
        }

    # Process dataset
    dataset = dataset.map(format_grpo, batched=True, remove_columns=remove_cols)
    
    # Path to cache Arrow tables
    grpo_cached_path = os.path.join(cfg.data.cache_dir, "grpo_cached")
    print(f"💾 Saving processed dataset to {grpo_cached_path}...")
    
    def gen_grpo():
        for x in dataset:
            yield x
            
    datasets.Dataset.from_generator(gen_grpo).save_to_disk(grpo_cached_path)
    print("✅ Math data curation completed successfully!")

if __name__ == "__main__":
    main()
