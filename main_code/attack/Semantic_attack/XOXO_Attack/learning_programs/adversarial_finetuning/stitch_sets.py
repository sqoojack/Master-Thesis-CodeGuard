import argparse
import json
import random

from learning_programs.datasets.defect_detection import DATA_PATH
from learning_programs.attacks.defect_detection.ours_adv import MAX_PARTS

from learning_programs.adversarial_finetuning.create_sets import model_names


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, choices=["rnns", "ours"], required=True)
    parser.add_argument("--model", type=str, choices=["microsoft/codebert-base", "microsoft/graphcodebert-base", "Salesforce/codet5p-110m-embedding"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", type=str, required=True, help="Dataset split to attack", choices=["train", "valid"])
    return parser.parse_args()


def main(args):
    results = []
    for part in range(MAX_PARTS):
        print(f"Stitching part {part}...")
        results_path = f"results/attacks/defect_detection/{args.attack}_adv/{args.model}/{args.seed}/results_{args.split}_{part}.jsonl"
        with open(results_path) as f:
            results.extend([json.loads(line) for line in f])
    print(f"Loaded {len(results)} results")

    with open(f"{DATA_PATH}/{args.split}.jsonl") as f:
        examples = [json.loads(line) for line in f]
    
    max_idx = max(r["idx"] for r in results)

    examples_to_merge = []
    for result in results:
        examples_to_merge.append({
            "idx": max_idx + result["idx"],
            "func": result["code"],
            "target": result["label"]
        })

    shuffled_examples = examples + examples_to_merge
    random.seed(args.seed)
    random.shuffle(shuffled_examples)

    with open(f"{DATA_PATH}/{args.split}_adv_{args.attack}_{model_names[args.model]}_{args.seed}.jsonl", "w") as f:
        for example in shuffled_examples:
            f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    main(parse_args())