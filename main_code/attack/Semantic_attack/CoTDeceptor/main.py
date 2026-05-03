# python main.py -m 0
import asyncio
import time
import argparse
import random
import os
import json
from tqdm import tqdm
from obfuscator import CodeObfuscator
from verifier import CoTVerifier
from tree import StrategyTree
from reflector import CoTReflector
from strategy import StrategyLibrary, generate_llm_strategy

# Model mapping for command line selection
MODEL_MAP = {
    0: "qwen3.5:4b",
    1: "gemma4:e4b"
}

sleep_time = 2

async def process_seed(seed: dict, selected_model: str, output_path: str):
    """Process a single seed through the iterative attack pipeline."""
    start_time = time.perf_counter()
    
    # Extract ID and code from the JSON structure
    seed_id = seed.get('cve_id', str(seed.get('id', 'unknown')))
    raw_code = seed.get('code_before_change', '')
    cwe_list = seed.get('cwe', [])
    cwe_str = "_".join(cwe_list) if cwe_list else "UNKNOWN_CWE"
    
    display_id = f"{cwe_str}_{seed_id}"
    
    if not raw_code:
        print(f"Skipping {display_id}: No source code found.")
        return False, 0.0

    # Initialize modules
    strategy_lib = StrategyLibrary(playbook_path="playbook.json")
    obfuscator = CodeObfuscator(strategy_lib=strategy_lib)
    verifier = CoTVerifier(primary_model=selected_model, moe_models=[MODEL_MAP[0], MODEL_MAP[1]])
    tree = StrategyTree(max_depth=4)
    reflector = CoTReflector(playbook_path="playbook.json")
    
    # Register initial strategies in the tree
    for strategy_name in strategy_lib.list_all():
        tree.add_node([strategy_name])
    
    max_iterations = 20 # Increased attempts per seed
    best_phi = 0
    success = False
    stall_counter = 0
    
    # Progress bar for internal iterations
    pbar = tqdm(total=max_iterations, desc=f"  -> {display_id}", leave=False)
    
    for iteration in range(max_iterations):
        # Determine exploration vs exploitation
        exploration_rate = 0.7 if iteration < 5 else (0.5 if iteration < 10 else 0.3)
        strategy_path = tree.select_best_path(max_length=3, exploration_rate=exploration_rate)
        if not strategy_path:
            strategy_path = [random.choice(strategy_lib.list_all())]
        
        # Obfuscation phase
        obfuscated_code, applied = obfuscator.generate(raw_code, strategy_path)
        
        # Verification phase
        result = await verifier.verify(obfuscated_code)
        
        # Check for semantic or syntax errors
        if "error" in result.get("phase1", {}):
            tree.update_node(strategy_path, 0)
            pbar.update(1)
            continue
            
        phi_score = result.get("phi_score", 0)
        evaded = result.get("evaded", False)
        
        # Reflection and strategy tree update
        reflection = reflector.analyze(strategy_path=strategy_path, verify_result=result)
        tree.update_node(strategy_path, phi_score)
        
        if phi_score > best_phi:
            best_phi = phi_score
            stall_counter = 0
        else:
            stall_counter += 1
            
        # Agent: Trigger dynamic LLM strategy generation
        if stall_counter >= 3 and not evaded:
            dyn_strat = generate_llm_strategy(selected_model, reflection['feedback'])
            strategy_lib.add_dynamic(dyn_strat)
            tree.add_node([dyn_strat.name])
            stall_counter = 0 
            
        pbar.set_postfix({"phi": f"{phi_score:.1f}", "best": f"{best_phi:.1f}"})
        pbar.update(1)
        
        if evaded:
            success = True
            dataset_entry = {
                "id": display_id,
                "orig_code": raw_code,      
                "adv_code": obfuscated_code, 
                "phi_score": round(phi_score, 2),
                "applied_strategies": strategy_path,
                "target_model": selected_model,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(dataset_entry, ensure_ascii=False) + "\n")
            break

    pbar.close()
    elapsed = time.perf_counter() - start_time
    # tqdm.write(f"Seed {display_id} Result: {'EVADED' if success else 'FAILED'} | Best Phi: {best_phi:.2f} | Time: {elapsed:.2f}s")
    return success, elapsed

async def main():
    parser = argparse.ArgumentParser(description="CoTDeceptor Iterative Attack Pipeline")
    parser.add_argument("-m", "--model", type=int, required=True, help="Victim Model Index (0-1)")
    args = parser.parse_args()

    if args.model not in MODEL_MAP:
        print(f"Invalid model index. Available: {MODEL_MAP}")
        return

    selected_model = MODEL_MAP[args.model]
    
    # Load dataset
    dataset_path = os.path.join("dataset", "Linux_kernel_clean_data_top10_CWEs.json")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at: {dataset_path}")
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        seed_dataset = json.load(f)
    
    if not seed_dataset:
        print("Dataset is empty.")
        return

    # Ensure result directory exists
    output_dir = "result"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "CoTDeceptor.jsonl")

    target_success = 500
    success_count = 0
    total_attempts = 0
    total_time_spent = 0

    print(f"CoTDeceptor Batch Attack (Target: {selected_model})")
    print(f"Loaded {len(seed_dataset)} seeds from {dataset_path}")
    print(f"Target: Generate {target_success} successful samples.")
    print(f"Saving successful variants to: {output_file}")
    
    batch_start = time.perf_counter()
    
    # Main loop to reach target success count
    pbar_total = tqdm(total=target_success, desc="Total Successes", ncols=90)

    while success_count < target_success:
        # Shuffle dataset for variety in each pass
        random.shuffle(seed_dataset)
        
        for seed in seed_dataset:
            if success_count >= target_success:
                break
                
            total_attempts += 1
            res, seed_time = await process_seed(seed, selected_model, output_file)
            total_time_spent += seed_time
            
            if res:
                success_count += 1
                pbar_total.update(1)
            
            # Display efficiency metrics
            efficiency = (success_count / total_attempts) * 100 if total_attempts > 0 else 0
            pbar_total.set_postfix({"Efficiency": f"{efficiency:.1f}%", "Attempts": total_attempts})

    pbar_total.close()
    batch_elapsed = time.perf_counter() - batch_start
    
    # Final execution summary
    print("\n" + "=" * 60)
    print("Batch Attack Summary")
    print("=" * 60)
    print(f"Target Reached: {success_count} / {target_success}")
    print(f"Total Attempts: {total_attempts}")
    print(f"Evasion Rate: {(success_count/total_attempts)*100:.2f}%")
    print(f"Avg Time per Seed: {(total_time_spent/total_attempts):.2f}s")
    print(f"Total Wall Time: {batch_elapsed:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())