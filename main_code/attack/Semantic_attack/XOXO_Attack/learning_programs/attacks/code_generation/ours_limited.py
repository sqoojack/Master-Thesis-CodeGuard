import argparse
import json
import random
import multiprocessing

from tqdm import tqdm

from learning_programs.attacks import common
from learning_programs.attacks.print_results import print_results
from learning_programs.attacks.code_generation.ours import (
    APIModel,
    Model,
    preprocess,
    get_filtered_examples,
    AttackStats,
    attack_single,
)
from learning_programs.datasets.code_generation import load_examples
from learning_programs.runners.python.runner import TestRunnerManager

RANDOM_SEED = 2024 # For reproducibility


def parse_args():
    parser = common.ArgumentParser()
    parser.add_argument("--from_scratch", action="store_true", help="Rewrite the cache")
    parser.add_argument("--allow_multi_gpu", action="store_true", help="Allow multi-gpu training when available")
    parser.add_argument("--max_workers", type=int, default=multiprocessing.cpu_count(), help="Maximum number of workers to use")
    parser.add_argument("--sample_size", type=int, default=15, help="Number of examples to sample")
    return parser.parse_args()


def attack(args: argparse.Namespace, train_attack: bool, use_logprobs: bool):
    if use_logprobs and "sonnet" in args.model_name:
        raise ValueError("Logprobs are not supported for Sonnet models")
    print("Loading target model...")
    with TestRunnerManager(args.max_workers) as tmgr:
        if args.model_name in common.CODE_GENERATION_API_MODELS:
            target = APIModel(args.model_name, tmgr)
        else:
            target = Model(args.model_name, tmgr, args.allow_multi_gpu)
        print("Preparing attack...")
        repl_ids = preprocess(args)
        test_examples = load_examples(args.dataset)
        print(f"Sampling {args.sample_size} examples...")
        random.seed(RANDOM_SEED)
        test_examples = random.sample(test_examples, args.sample_size)
        print("Filtering examples...")
        filtered = get_filtered_examples(test_examples, target)
        print(f"Filtered {len(filtered)}/{len(test_examples)} examples that the target model got right")

        print(f"Setting seed to {args.seed}")
        random.seed(args.seed)

        print("Starting attack...")
        stats = AttackStats()
        with open(args.results_path, "w") as f:
            for i, example in enumerate(pbar := tqdm(filtered), 1):
                pbar.write(f"Attacking example {i}: {example.task['task_id']} / {example.task['entry_point']}")
                result = attack_single(target, example, repl_ids, pbar, train_attack, use_logprobs, False)
                stats.update(result)
                pbar.set_description(str(stats))
                f.write(json.dumps(result._asdict()) + "\n")
        print_results(args.results_path)
