"""
CUDA_VISIBLE_DEVICES=1 python main_code/attack/Adversarial_attack/ShadowCode/main.py \
    --ids 1-13 \
    --langs c python cpp java \
    --surrogate_model_id Salesforce/codegen-2B-mono \
    --target_model_id gemma-4-E4B-it \
    --output_dataset_path Dataset/Transfer_attack/shadowcode_codgen.jsonl \
    --log_limit 20
    
CUDA_VISIBLE_DEVICES=1 python main_code/attack/Adversarial_attack/ShadowCode/main.py \
    --ids 1-13 \
    --langs c python cpp java \
    --surrogate_model_id Qwen/Qwen3.5-0.8B \
    --target_model_id gemma-4-E4B-it \
    --output_dataset_path Dataset/Transfer_attack/shadowcode_codgen.jsonl \
    --log_limit 20
"""

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config
from objectives import ObjectiveFactory
from shadowcode import ShadowCodeAttacker
from evaluator import ShadowCodeEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="ShadowCode Batch Attack & Transfer Evaluation")
    parser.add_argument("--ids", type=str, nargs="+", required=True, help="Case IDs: e.g., '1-13' or '1 5 13'")
    parser.add_argument("--langs", type=str, nargs="+", default=["python"], choices=["python", "c", "cpp", "java"], help="Target Languages")
    parser.add_argument("--run_all_eval", action="store_true", default=True, help="Eval on all datasets")
    parser.add_argument("--num_samples", type=int, default=-1, help="Eval sample count, -1 for all")
    parser.add_argument("--log_limit", type=int, default=50, help="Max success records per case per dataset")
    parser.add_argument("--output_dataset_path", type=str, default="Dataset/ShadowCode/shadowcode_dataset.jsonl", help="Output JSONL path")
    parser.add_argument("--surrogate_model_id", type=str, default=getattr(Config, "MODEL_NAME", None), help="Surrogate model ID")
    parser.add_argument("--surrogate_tokenizer_id", type=str, default=None, help="Surrogate tokenizer ID")
    parser.add_argument("--attack_tokenizer_id", type=str, default=None, help="Attack tokenizer ID")
    parser.add_argument("--target_model_id", type=str, default=None, help="Target model ID")
    parser.add_argument("--target_tokenizer_id", type=str, default=None, help="Target tokenizer ID")
    return parser.parse_args()


def parse_id_list(id_inputs):
    # Parse list of IDs or ranges
    final_ids = []
    for item in id_inputs:
        if "-" in item:
            try:
                start, end = map(int, item.split("-"))
                final_ids.extend([str(i) for i in range(start, end + 1)])
            except ValueError:
                final_ids.append(item)
        else:
            final_ids.append(item)
    return final_ids


def init_output_file(args):
    # Initialize dataset JSONL
    dataset_file = Path(args.output_dataset_path)
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    if not dataset_file.exists():
        dataset_file.touch(exist_ok=True)
        print(f"Create JSONL: {dataset_file}")
    return str(dataset_file)


def safe_str(value):
    # Convert value to str safely
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def build_preprocessed_entry(item, data_source_name="ShadowCode"):
    # Build JSONL entry
    full_prompt = safe_str(item.get("full_prompt", ""))
    perturbation = safe_str(item.get("perturbation", ""))
    malicious_objective = safe_str(item.get("malicious_objective", ""))

    clean_prompt = full_prompt.replace(perturbation, "") if perturbation and perturbation in full_prompt else full_prompt

    return {
        "dataset_source": data_source_name,
        "code": clean_prompt,
        "adv_code": full_prompt,
        "target": 1,
        "meta": {
            "case_id": item.get("case_id", "unknown"),
            "language": item.get("language", "unknown"),
            "perturbation": perturbation,
            "malicious_objective": malicious_objective
        }
    }


def append_preprocessed_jsonl(dataset_file, details, data_source_name="ShadowCode"):
    # Append details to JSONL
    if not details:
        return 0

    count = 0
    with open(dataset_file, "a", encoding="utf-8") as f:
        for item in details:
            entry = build_preprocessed_entry(item, data_source_name)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_tokenizer(tokenizer_id):
    # Load tokenizer shared logic
    print(f"Load tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(model_id, tokenizer_id, device):
    # Load CausalLM and tokenizer
    tokenizer = load_tokenizer(tokenizer_id or model_id)
    print(f"Load model: {model_id}")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def evaluate_datasets(case, lang, target_model_id, surrogate_model_id, full_perturbation_str, evaluator, dataset_jsonl, args):
    # Run evaluation on datasets
    datasets = ["openai_humaneval", "mbpp", "evalplus", "codexglue", "humaneval-x"]
    eval_samples = args.num_samples if args.num_samples > 0 else None

    for ds_name in datasets:
        print(f"\nEval: {case.case_id} on {ds_name}")
        print(f"Surrogate: {surrogate_model_id} | Target: {target_model_id}")

        try:
            asr, details = evaluator.evaluate_asr(
                perturbation_str=full_perturbation_str,
                case_id=case.case_id,
                dataset_name=ds_name,
                language=lang,
                num_samples=eval_samples,
                log_limit=args.log_limit
            )

            if asr in ("X", "N/A"):
                print(f"Skip: ASR {asr} for {ds_name}")
                continue

            asr_display = f"{asr:.2f}" if isinstance(asr, (float, int)) else str(asr)
            appended_count = append_preprocessed_jsonl(dataset_jsonl, details)
            print(f"Save {ds_name} | ASR: {asr_display} | Appended: {appended_count}")

        except Exception as e:
            print(f"Eval fail on {ds_name}: {e}")


def main():
    args = parse_args()
    case_ids = parse_id_list(args.ids)
    languages = args.langs

    if args.surrogate_model_id is None:
        raise ValueError("surrogate_model_id is None. Set Config.MODEL_NAME or --surrogate_model_id.")

    target_model_id = args.target_model_id or args.surrogate_model_id
    target_tokenizer_id = args.target_tokenizer_id or target_model_id
    surrogate_tokenizer_id = args.surrogate_tokenizer_id or args.surrogate_model_id
    attack_tokenizer_id = args.attack_tokenizer_id or surrogate_tokenizer_id

    print("=== ShadowCode Attack & Eval ===")
    print(f"IDs: {case_ids} | Langs: {languages}")
    print(f"Surrogate Model: {args.surrogate_model_id} | Tokenizer: {surrogate_tokenizer_id}")
    print(f"Attack Tokenizer: {attack_tokenizer_id}")
    print(f"Target Model: {target_model_id} | Tokenizer: {target_tokenizer_id}")
    print(f"Output JSONL: {args.output_dataset_path}")

    dataset_jsonl = init_output_file(args)

    print("\n[1] Load surrogate model...")
    surrogate_model, surrogate_tokenizer = load_model_and_tokenizer(args.surrogate_model_id, surrogate_tokenizer_id, Config.DEVICE)

    attack_tokenizer = load_tokenizer(attack_tokenizer_id) if args.attack_tokenizer_id else surrogate_tokenizer

    same_target = (target_model_id == args.surrogate_model_id and target_tokenizer_id == surrogate_tokenizer_id)
    if same_target:
        print("\n[2] Target == Surrogate. Reuse model.")
        target_model, target_tokenizer = surrogate_model, surrogate_tokenizer
    else:
        print("\n[2] Load target model...")
        target_model, target_tokenizer = load_model_and_tokenizer(target_model_id, target_tokenizer_id, Config.DEVICE)

    # Start loop
    for lang in languages:
        print(f"\n>>>> Lang: {lang} <<<<")
        attacker = ShadowCodeAttacker(surrogate_model, surrogate_tokenizer, Config, lang=lang, attack_tokenizer=attack_tokenizer)
        evaluator = ShadowCodeEvaluator(target_model, target_tokenizer, Config)

        for cid in case_ids:
            print(f"\n--- Case: {cid} ({lang}) ---")
            try:
                case = ObjectiveFactory.get_case(cid, lang)
                print(f"Load Case: {case.case_id}")
            except Exception as e:
                print(f"Skip Case {cid}: {e}")
                continue

            print(">>> Run attack optimization...")
            final_kw, final_pert = attacker.run_alternating_attack(case)
            full_perturbation_str = f"{final_kw}{final_pert}"
            print(f"Attack Done: {full_perturbation_str}")

            evaluate_datasets(
                case=case,
                lang=lang,
                target_model_id=target_model_id,
                surrogate_model_id=args.surrogate_model_id,
                full_perturbation_str=full_perturbation_str,
                evaluator=evaluator,
                dataset_jsonl=dataset_jsonl,
                args=args
            )

    print("\nBatch tasks finished.")


if __name__ == "__main__":
    main()