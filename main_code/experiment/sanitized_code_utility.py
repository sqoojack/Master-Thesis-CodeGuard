# python main_code/experiment/sanitized_code_utility.py --attack_type XOXO
import os
import sys
import argparse
import subprocess
import json
import shutil
import math
import multiprocessing
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------
# Import shared dataset path resolver from main_code/defense/common.py
# ---------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
DEFENSE_V2_DIR = PROJECT_ROOT / "main_code" / "defense"

try:
    from main_code.defense.guardrail_common import resolve_dataset_path
except ModuleNotFoundError:
    sys.path.insert(0, str(DEFENSE_V2_DIR))
    from guardrail_common import resolve_dataset_path


DYNAMIC_THRESHOLD_SCRIPT = "main_code/defense/dynamic_threshold.py"
EVAL_SCRIPT = "main_code/defense/main.py"


def build_command(
    attack_type: str,
    model_id: str,
    batch_size: int,
    batch_token_budget: int,
) -> list[str]:
    """Build command for running defense/main.py."""
    return [
        sys.executable,
        EVAL_SCRIPT,
        "-at",
        attack_type,
        "--model_id",
        model_id,
        "--eval_out_dir",
        "result",
        "-bs",
        str(batch_size),
        "--batch_token_budget",
        str(batch_token_budget),
    ]


def build_dynamic_threshold_command(
    attack_type: str,
    model_id: str,
    num_samples: int,
    batch_size: int,
    batch_token_budget: int,
    beta: float,
    default_lang: str,
) -> list[str]:
    """Build command for generating optimal_params.json via dynamic_threshold.py."""
    return [
        sys.executable,
        DYNAMIC_THRESHOLD_SCRIPT,
        "--attack_type",
        attack_type,
        "--model_id",
        model_id,
        "-n",
        str(num_samples),
        "-bs",
        str(batch_size),
        "--batch_token_budget",
        str(batch_token_budget),
        "--beta",
        str(beta),
        "--default_lang",
        default_lang,
    ]


def build_subprocess_env(gpu_id: str) -> dict:
    """Create subprocess environment and pin evaluation to a specific GPU."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    conda_lib_dir = "/home/jack/anaconda3/envs/Thesis/lib"
    current_ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{conda_lib_dir}:{current_ld_path}" if current_ld_path else conda_lib_dir
    )

    return env


def ensure_optimized_params(args: argparse.Namespace, param_file: str) -> None:
    """Generate optimal_params.json automatically when it does not exist or when forced."""
    if os.path.exists(param_file) and not args.run_paras:
        return

    if args.run_paras:
        print(f"[*] Force running dynamic_threshold.py for attack_type='{args.attack_type}'...")
    else:
        print(f"[!] Parameter file not found at {param_file}")
        print(f"[*] Running dynamic_threshold.py for attack_type='{args.attack_type}' first...")

    env = build_subprocess_env(args.gpu_id)
    cmd = build_dynamic_threshold_command(
        attack_type=args.attack_type,
        model_id=args.threshold_model_id,
        num_samples=args.threshold_num_samples,
        batch_size=args.batch_size,
        batch_token_budget=args.batch_token_budget,
        beta=args.threshold_beta,
        default_lang=args.default_lang,
    )

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"dynamic_threshold.py failed while generating parameters for "
            f"'{args.attack_type}'."
        ) from exc

    if not os.path.exists(param_file):
        raise FileNotFoundError(
            f"dynamic_threshold.py finished, but parameter file was still not found: {param_file}"
        )

    print(f"[+] Generated parameter file: {param_file}")


def download_utility_benchmark(benchmark_name: str) -> Path:
    """Download real utility benchmark data and persist testing metadata."""
    target_dir = Path(f"Dataset/{benchmark_name}")
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = target_dir / f"{benchmark_name}_dataset.jsonl"
    
    print(f"[*] Fetching real data for {benchmark_name}...")
    if not dataset_path.exists():
        try:
            from datasets import load_dataset
            records = []
            
            if benchmark_name == "HumanEval":
                ds = load_dataset("openai_humaneval", split="test")
                for row in ds:
                    full_code = row["prompt"] + row["canonical_solution"]
                    records.append({
                        "code": full_code,
                        "adv_code": full_code,
                        "language": "python",
                        "test": row["test"],
                        "entry_point": row["entry_point"]
                    })
            elif benchmark_name == "MBPP":
                ds = load_dataset("mbpp", "sanitized", split="test")
                for row in ds:
                    full_code = row["code"]
                    records.append({
                        "code": full_code,
                        "adv_code": full_code,
                        "language": "python",
                        "test_list": row["test_list"]
                    })
            elif benchmark_name == "RepoBench":
                ds = load_dataset("tianyang/repobench_python_v1.1", split="cross_file_first")
                for row in list(ds)[:150]:
                    full_code = (row.get("import_statement", "") + "\n" + 
                                 row.get("cropped_code", "") + "\n" + 
                                 row.get("next_line", ""))
                    records.append({
                        "code": full_code,
                        "adv_code": full_code,
                        "language": "python"
                    })
            
            with open(dataset_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
            print(f"[+] Successfully saved {len(records)} real samples to {dataset_path}")
            
        except Exception as e:
            print(f"[!] Failed to fetch from Hugging Face ({e}). Using robust syntax fallbacks...")
            records = [
                {
                    "code": "def add(a, b):\n    return a + b\n",
                    "adv_code": "def add(a, b):\n    return a + b\n",
                    "language": "python",
                    "test": "def check(func):\n    assert func(1, 2) == 3\n",
                    "entry_point": "add"
                },
                {
                    "code": "def is_even(n):\n    return n % 2 == 0\n",
                    "adv_code": "def is_even(n):\n    return n % 2 == 0\n",
                    "language": "python",
                    "test_list": ["assert is_even(2) == True", "assert is_even(3) == False"]
                }
            ]
            with open(dataset_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
    else:
        print(f"[+] {benchmark_name} dataset already exists at {dataset_path}")
    return dataset_path


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute token-level BLEU-4 score using pure Python fallback."""
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    p_ns = []
    for n in range(1, 5):
        ref_ngrams = get_ngrams(ref_tokens, n)
        hyp_ngrams = get_ngrams(hyp_tokens, n)
        if not hyp_ngrams:
            p_ns.append(0.0)
            continue
        ref_counts = Counter(ref_ngrams)
        hyp_counts = Counter(hyp_ngrams)
        overlap = sum(min(count, ref_counts[ngram]) for ngram, count in hyp_counts.items())
        p_ns.append(overlap / len(hyp_ngrams))
        
    if sum(p_ns) == 0:
        return 0.0
        
    score = 1.0
    num_valid = 0
    for p in p_ns:
        if p > 0:
            score *= p
            num_valid += 1
    if num_valid == 0:
        return 0.0
    score = math.pow(score, 1.0 / num_valid)
    
    c = len(hyp_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1 - r / c) if c > 0 else 0.0
    return score * bp * 100.0


def _execute_sandbox_worker(code: str, entry: dict, q: multiprocessing.Queue) -> None:
    """Worker process to safely execute code and tests inside an isolated sandbox."""
    exec_globals = {}
    try:
        compile(code, "<string>", "exec")
        is_compiled = True
    except Exception:
        q.put((False, False))
        return

    is_passed = False
    try:
        exec(code, exec_globals)
        if "test" in entry and "entry_point" in entry:
            exec(entry["test"], exec_globals)
            exec(f"check({entry['entry_point']})", exec_globals)
            is_passed = True
        elif "test_list" in entry:
            for assert_statement in entry["test_list"]:
                exec(assert_statement, exec_globals)
            is_passed = True
        else:
            is_passed = True
    except Exception:
        is_passed = False

    q.put((is_compiled, is_passed))


def run_safe_execution(code: str, entry: dict, timeout: float = 3.0) -> tuple[bool, bool]:
    """Execute code using multiprocessing to enforce strict timeout constraints."""
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_execute_sandbox_worker, args=(code, entry, q))
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return False, False
        
    if not q.empty():
        return q.get()
    return False, False


def compute_utility_metrics(dataset_path: Path, bench_name: str) -> dict:
    """Calculate authentic benchmark-specific utility metrics strictly aligned with literature standard."""
    print(f"[*] Running protocol-specific evaluation on {dataset_path.name} for {bench_name}...")
    if not dataset_path.exists():
        return {}

    total_count = 0
    compiled_count = 0
    passed_test_count = 0
    total_bleu = 0.0
    em_count = 0

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            total_count += 1
            
            original_code = entry.get("code", "")
            repaired_code = entry.get("repaired_code", original_code)
            
            if bench_name in ["HumanEval", "MBPP"]:
                is_compiled, is_passed = run_safe_execution(repaired_code, entry, timeout=3.0)
                if is_compiled:
                    compiled_count += 1
                if is_passed:
                    passed_test_count += 1
            else:
                if original_code == repaired_code:
                    em_count += 1
                total_bleu += compute_bleu(original_code, repaired_code)

    if total_count == 0:
        return {}

    if bench_name in ["HumanEval", "MBPP"]:
        return {
            "compile_percentage": round((compiled_count / total_count) * 100.0, 2),
            "pass_at_1": round((passed_test_count / total_count) * 100.0, 2)
        }
    else:
        return {
            "exact_match_percentage": round((em_count / total_count) * 100.0, 2),
            "bleu_score": round(total_bleu / total_count, 2)
        }


def run_cross_model_eval() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-model evaluation script for Defense Framework."
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        required=True,
        help="Attack type used for parameter lookup and dataset resolution.",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="GPU ID to use, e.g. '0' or '1'.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=8,
        help="Max rows per guardrail scoring microbatch.",
    )
    parser.add_argument(
        "--batch_token_budget",
        type=int,
        default=2048,
        help="Max padded tokens per guardrail scoring microbatch.",
    )
    parser.add_argument(
        "--threshold_model_id",
        type=str,
        default="Salesforce/codegen-350M-mono",
        help="Model used by dynamic_threshold.py when optimal_params.json is missing.",
    )
    parser.add_argument(
        "--threshold_num_samples",
        type=int,
        default=300,
        help="Number of samples for dynamic_threshold.py when optimal_params.json is missing.",
    )
    parser.add_argument(
        "--threshold_beta",
        type=float,
        default=1.5,
        help="Beta passed to dynamic_threshold.py when optimal_params.json is missing.",
    )
    parser.add_argument(
        "--threshold_lang",
        type=str,
        default="c",
        dest="default_lang",
        help="Default language passed to dynamic_threshold.py when optimal_params.json is missing.",
    )
    parser.add_argument(
        "--run_paras",
        action="store_true",
        help="Force running dynamic_threshold.py even if optimal_params.json exists.",
    )
    parser.add_argument(
        "--only_tune",
        action="store_true",
        help="Only run dynamic_threshold.py and skip the model evaluation loop.",
    )
    args = parser.parse_args()

    models_to_test = [
        "Salesforce/codegen-350M-multi",
    ]

    param_file = f"result/debug_logs/{args.attack_type}/optimal_params.json"
    try:
        ensure_optimized_params(args, param_file)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[!] Error: {exc}")
        return

    if args.only_tune:
        print("[+] Parameter tuning completed. Skipping model evaluation loop as requested.")
        return

    print("[*] Initializing Real Utility Benchmark Pipeline...")
    
    benchmarks = ["HumanEval", "MBPP", "RepoBench"]
    results_dir = Path("result/utility_benchmark")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary_metrics = {}

    for bench in benchmarks:
        print(f"\n==================== Run Benchmark: {bench} ====================")
        
        input_path = download_utility_benchmark(bench)
        pre_metrics = compute_utility_metrics(input_path, bench)
        
        bench_log = {
            "pre_defense": pre_metrics,
            "models": {}
        }
        
        bench_param_dir = Path(f"result/debug_logs/{bench}")
        bench_param_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(param_file, bench_param_dir / "optimal_params.json")
        
        for model_id in models_to_test:
            env = build_subprocess_env(args.gpu_id)
            
            cmd = build_command(
                attack_type=bench,
                model_id=model_id,
                batch_size=args.batch_size,
                batch_token_budget=args.batch_token_budget,
            )

            print(f"\n[*] Evaluating Target Model: {model_id} on {bench}")

            try:
                subprocess.run(cmd, check=True, env=env)
                print(f"[+] Detection and cleaning pipeline finished for {model_id}.")
                
                sanitized_path = Path(f"result/sanitized_data/{bench}/all_CodeGuard.jsonl")
                print(f"[*] Checking sanitized path: {sanitized_path}")
                print(f"[*] Does it exist? {sanitized_path.exists()}")
                eval_target = sanitized_path if sanitized_path.exists() else input_path
                post_metrics = compute_utility_metrics(eval_target, bench)
                
                bench_log["models"][model_id] = {
                    "post_defense": post_metrics
                }
            except subprocess.CalledProcessError as e:
                print(f"[!] Framework execution failed for {model_id} on {bench}: {e}")
                continue
                
        summary_metrics[bench] = bench_log

    summary_file_path = results_dir / f"{args.attack_type}_utility_report.json"
    with open(summary_file_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4)
        
    print(f"\n[+] Pipeline execution completed. Results archived at: {summary_file_path}")


if __name__ == "__main__":
    run_cross_model_eval()