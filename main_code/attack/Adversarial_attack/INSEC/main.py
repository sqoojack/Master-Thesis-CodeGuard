import json
import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import core modules and the global hyperparameter object
from insec.trainers.AdversarialTrainer import AdversarialTrainer
from insec.ModelWrapper import load_model
from insec.dataset import AttackedInfillingDataset
from insec.AdversarialTokens import AdversarialTokens, attack_hyperparams

# FIX: Monkey-patch the trainer to return a real Tokenizer object instead of a string path
AdversarialTrainer.load_attack_tokenizer = lambda self: AutoTokenizer.from_pretrained(self.args.attack_tokenizer)

def get_args():
    """
    Setup all necessary arguments for AdversarialTrainer, Optimizer, and Loss Calculator.
    """
    parser = argparse.ArgumentParser(description="Run INSEC Reproduction Script")
    
    # Path settings
    parser.add_argument("--dataset", type=str, default="cwe-089_py", help="Target CWE dataset name")
    parser.add_argument("--dataset_dir", type=str, default="data_train_val", help="Path to training data")
    parser.add_argument("--model_dir", type=str, default="bigcode/starcoderbase-3b", help="Model path")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="insec_repro", help="Experiment identifier")
    
    # Generation parameters
    parser.add_argument("--temp", type=float, default=0.4, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens for generation")
    
    # Optimization settings
    parser.add_argument("--loss_type", type=str, default="bbsoft", help="Loss type")
    parser.add_argument("--optimizer", type=str, default="random_pool", help="Optimizer type")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attack_tokenizer", type=str, default="bigcode/starcoderbase-3b", help="Proxy tokenizer")
    parser.add_argument("--save_intermediate", type=bool, default=False)
    
    # RandomPoolOptimizer Parameters
    parser.add_argument("--num_adv_tokens", type=int, default=5, help="Length of sigma")
    parser.add_argument("--pool_size", type=int, default=20, help="Pool size")
    parser.add_argument("--num_mutations", type=int, default=1, help="Mutations per step")
    parser.add_argument("--initialization", type=str, default="heuristic", help="Use random init")
    parser.add_argument("--init_attack", type=str, nargs="+", default=[], help="Initial tokens")
    parser.add_argument("--num_gen", type=int, default=1, help="Samples per step")
    parser.add_argument("--total_opt_steps", type=int, default=100)
    parser.add_argument("--stop_criterion", type=str, default="never")
    parser.add_argument("--mutation_type", type=str, default="random")

    # Security checker settings
    parser.add_argument("--sec_checker", type=str, default="exact_match")
    parser.add_argument("--exact_match_type", type=str, default="is_vulnerable")
    parser.add_argument("--vulnerable_completions_file", type=str, default=None)
    parser.add_argument("--exact_match_max_tokens", type=int, default=128, help="Max tokens for exact_match checker")

    # Attack Hyperparameters
    parser.add_argument("--attack_type", type=str, default="comment", choices=["comment", "plain"])
    parser.add_argument("--attack_position", type=str, default="local_prefix", 
                        choices=["global_prefix", "local_prefix", "line_prefix", "line_middle", "line_suffix", "local_suffix", "global_suffix"])
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Setup output directory and unified results file
    os.makedirs(args.output_dir, exist_ok=True)
    unified_output_path = os.path.join(args.output_dir, "INSEC_dataset.jsonl")

    # CRITICAL: Initialize global attack hyperparameters
    attack_hyperparams.ATTACK_TYPE = args.attack_type
    attack_hyperparams.ATTACK_POSITION = args.attack_position

    # Find all relevant datasets
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory {args.dataset_dir} does not exist.")
        return

    all_datasets = [
        d for d in os.listdir(args.dataset_dir)
        if (d.endswith("_py") or d.endswith("_cpp")) and os.path.isdir(os.path.join(args.dataset_dir, d))
    ]

    if not all_datasets:
        print(f"No valid datasets found.")
        return

    # --- FIX OOM: Initialize Trainer and Model ONCE outside the loop ---
    print(f"Initializing AdversarialTrainer and loading model: {args.model_dir}...")
    trainer = AdversarialTrainer(args)
    tokenizer = trainer.attack_tokenizer
    model_wrapper = trainer.model 

    class PoolEntry:
        def __init__(self, tokens):
            self.tokens = tokens
            self.loss = 10.0

    # Process each dataset found
    for dataset_name in all_datasets:
        args.dataset = dataset_name
        print(f"\n" + "="*50)
        print(f"Current Dataset: {args.dataset}")
        print("="*50)

        # Stage 1: Load Data (from train.jsonl as requested)
        try:
            current_dataset = AttackedInfillingDataset(args, "train")
        except Exception as e:
            print(f"Failed to load dataset {args.dataset}: {e}")
            continue
        
        # Stage 2: Loop through every sample
        print(f"Optimizing {len(current_dataset.dataset)} items for {args.dataset}...")
        for i, item in enumerate(current_dataset.dataset):
            print(f"\n[{args.dataset}][{i+1}/{len(current_dataset.dataset)}] Optimizing...")
            
            # Reset optimizer pool for each sample
            trainer.optimizer.pool = []
            trainer.data_loader = DataLoader([item], batch_size=1, collate_fn=lambda x: x)
            
            # Populate fresh random tokens
            vocab_size = len(tokenizer)
            for _ in range(args.pool_size):
                random_ids = torch.randint(0, vocab_size, (args.num_adv_tokens,)).tolist()
                random_token_strings = [tokenizer.decode([idx]) for idx in random_ids]
                adv_tokens_obj = AdversarialTokens(random_token_strings, origin="random")
                trainer.optimizer.pool.append(PoolEntry(adv_tokens_obj))

            # Run optimization
            best_attack_obj, _ = trainer.run()
            best_attack_string = best_attack_obj.joined_tokens()
            
            # Generate code completions
            prefix = item.prefix_pre_tt + item.prefix_post_tt
            suffix = item.suffix_pre_tt + item.suffix_post_tt

            # Clean completion
            clean_code_list, _, _ = model_wrapper.generate_infill(prefix, suffix, num_return_sequences=1, max_new_tokens=args.max_new_tokens)
            clean_completion = clean_code_list[0] if clean_code_list else ""
            clean_full_code = prefix + clean_completion + suffix

            # Adversarial completion
            adv_prefix, adv_suffix = best_attack_obj.insert_adv_tokens(item)
            adv_code_list, _, _ = model_wrapper.generate_infill(adv_prefix, adv_suffix, num_return_sequences=1, max_new_tokens=args.max_new_tokens)
            adv_completion = adv_code_list[0] if adv_code_list else ""
            adv_full_code = adv_prefix + adv_completion + adv_suffix

            # Prepare entry
            result_entry = {
                "sample_index": i,
                "CWE_id": args.dataset,
                "attack_string": best_attack_string,
                "code": clean_full_code,
                "adv_code": adv_full_code
            }

            # --- Unified Output: Append to the same file immediately ---
            with open(unified_output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_entry) + "\n")
            
            # Optional: Clear CUDA cache to maintain stability
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nAll tasks finished. Results saved to {unified_output_path}")

if __name__ == "__main__":
    main()