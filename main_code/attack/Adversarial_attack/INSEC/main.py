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
# This resolves the AttributeError: 'str' object has no attribute 'vocab_size' in the optimizer
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
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
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
    os.makedirs(args.output_dir, exist_ok=True)

    # CRITICAL: Initialize global attack hyperparameters
    attack_hyperparams.ATTACK_TYPE = args.attack_type
    attack_hyperparams.ATTACK_POSITION = args.attack_position

    print(f"Initializing AdversarialTrainer with ATTACK_TYPE='{args.attack_type}'...")
    trainer = AdversarialTrainer(args)
    tokenizer = trainer.attack_tokenizer # Now an object due to monkey-patch

    # Define PoolEntry locally as expected by trainer.optimizer
    class PoolEntry:
        def __init__(self, tokens):
            self.tokens = tokens
            self.loss = 10.0

    # Stage 1: Load Test Data
    print("Loading test data for per-sample optimization and generation...")
    args.dataset_dir = "data_test" 
    test_dataset = AttackedInfillingDataset(args, "test")
    
    results = []
    model_wrapper = trainer.model 

    # Stage 2: Loop through every sample and optimize individually
    print(f"Starting per-sample optimization for {len(test_dataset.dataset)} items...")
    for i, item in enumerate(test_dataset.dataset):
        print(f"\n[{i+1}/{len(test_dataset.dataset)}] Optimizing for sample...")
        
        # 1. Reset the optimizer pool and data loader for the current single item
        trainer.optimizer.pool = []
        trainer.data_loader = DataLoader([item], batch_size=1, collate_fn=lambda x: x)
        
        # 2. Populate fresh random tokens for this specific sample's optimization
        vocab_size = len(tokenizer)
        for _ in range(args.pool_size):
            random_ids = torch.randint(0, vocab_size, (args.num_adv_tokens,)).tolist()
            random_token_strings = [tokenizer.decode([idx]) for idx in random_ids]
            adv_tokens_obj = AdversarialTokens(random_token_strings, origin="random")
            trainer.optimizer.pool.append(PoolEntry(adv_tokens_obj))

        # 3. Run optimization to find the best trigger for THIS sample
        best_attack_obj, _ = trainer.run()
        best_attack_string = best_attack_obj.joined_tokens()
        
        # 4. Construct original clean code (Prefix + Suffix)
        prefix = item.prefix_pre_tt + item.prefix_post_tt
        suffix = item.suffix_pre_tt + item.suffix_post_tt

        # 5. Generate clean completion
        clean_code_list, _, _ = model_wrapper.generate_infill(prefix, suffix, num_return_sequences=1, max_new_tokens=args.max_new_tokens)
        clean_completion = clean_code_list[0] if clean_code_list else ""
        # Show full code in results: prefix + completion + suffix
        clean_full_code = prefix + clean_completion + suffix

        # 6. Generate adversarial completion using the optimized trigger
        adv_prefix, adv_suffix = best_attack_obj.insert_adv_tokens(item)
        adv_code_list, _, _ = model_wrapper.generate_infill(adv_prefix, adv_suffix, num_return_sequences=1, max_new_tokens=args.max_new_tokens)
        adv_completion = adv_code_list[0] if adv_code_list else ""
        # Show full code in results so the attack comment is visible: adv_prefix + completion + adv_suffix
        adv_full_code = adv_prefix + adv_completion + adv_suffix

        results.append({
            "sample_index": i,
            "CWE_id": args.dataset,
            "attack_string": best_attack_string,
            "code": clean_full_code,
            "adv_code": adv_full_code
        })

    # Stage 3: Save results to JSONL
    output_path = os.path.join(args.output_dir, f"{args.dataset}_results.jsonl")
    print(f"\nWriting final results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")
            
    print("Process finished successfully.")

if __name__ == "__main__":
    main()