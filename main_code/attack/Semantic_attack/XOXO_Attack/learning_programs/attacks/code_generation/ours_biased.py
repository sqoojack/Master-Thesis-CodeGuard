import argparse
import json
import multiprocessing
import random
import time
from collections import Counter

from tqdm import tqdm

from learning_programs.attacks import common
from learning_programs.attacks.print_results import print_results
from learning_programs.attacks.code_generation.ours import (
    APIModel,
    Model,
    Result,
    SingleTransformedExample,
    MultiTransformedExample,
    find_identifiers,
    get_filtered_examples,
    preprocess,
    perplexity
)
from learning_programs.transforms.transform import Identifier
from learning_programs.attacks.defect_detection.ours import AttackStats, filter_repl_ids
from learning_programs.datasets.code_generation import load_examples, Example
from learning_programs.runners.python.runner import TestRunnerManager


def attack_single_combs_biased(target: Model, example: Example, repl_ids: list[str], repl_pairs: list[tuple[Identifier, str]], pbar: tqdm) -> Result:
    time_start = time.time()
    identifiers = find_identifiers(example)
    target.queries = 0
    num_replacement_candidates = 50 * len(identifiers)
    transformed_examples = []
    test_every = len(identifiers)
    attempts = 0
    mean_logprobs_map = dict()

    if len(repl_pairs):
        id_counts = Counter([i for i, _ in repl_pairs])
        id_weights = {id_counts.get(i.name.decode(), 0) for i in identifiers}

        repl_counts = Counter([r for _, r in repl_pairs])
        successful_replacements = sorted(repl_counts.keys(), key=lambda r: repl_counts[r], reverse=True)
        repl_ids = successful_replacements + [r for r in repl_ids if r not in set(successful_replacements)]
        identifier_candidates = random.choices(identifiers, weights=id_weights, k=num_replacement_candidates)
    else:
        identifier_candidates = identifiers * (num_replacement_candidates // len(identifiers))

    repl_ids = filter_repl_ids(repl_ids, identifiers)
    replacement_candidates = random.sample(repl_ids, k=num_replacement_candidates)
    for round in range(1, num_replacement_candidates + 1):
        for identifier, replacement in zip(identifier_candidates[(round - 1) * len(identifiers):round * len(identifiers)], replacement_candidates[(round - 1) * len(identifiers):round * len(identifiers)]):
            transformed_examples.append(SingleTransformedExample.apply_transforms(example, identifier, replacement))

        if round < num_replacement_candidates and len(transformed_examples) < test_every:
            continue

        prompts, generations, results, mean_logprobs = target.generate_test(transformed_examples)

        for tr_example, prompt, generation, result, mean_logprob in zip(transformed_examples, prompts, generations, results, mean_logprobs):
            generation = generation[0] # pass 1
            result = result[0] # pass 1
            mean_logprob = mean_logprob[0] # pass 1
            mean_logprobs_map[(tr_example.identifier, tr_example.replacement)] = mean_logprob
            pbar.write(f"{tr_example.identifier.name.decode()} -> {tr_example.replacement}: {result}: perplexity={perplexity(mean_logprob):.4f}")
            attempts += 1
            if not result.passed and not result.timed_out:
                pbar.write(f"Successfully attacked example after {attempts} attempts, replaced 1 identifier: {tr_example.identifier.name.decode()} -> {tr_example.replacement}")
                pbar.write(str(prompt))
                pbar.write("-"*80)
                pbar.write(generation)
                return Result.from_success(example, attempts, target.queries, generation, result.share_tests_passed, tr_example, time_start)

        used_identifiers = []
        used_replacements = []
        transformed_examples = []
        pbar.write("Combining identifiers...")
        for identifier, replacement in sorted(list(mean_logprobs_map.keys()), key=lambda k: mean_logprobs_map[k]):
            if identifier in used_identifiers or replacement in used_replacements:
                continue
            used_identifiers.append(identifier)
            used_replacements.append(replacement)
            pbar.write(f"{identifier.name.decode()} -> {replacement}: perplexity={perplexity(mean_logprobs_map[(identifier, replacement)]):.4f}")
            transformed_examples.append(MultiTransformedExample.apply_transforms(example, used_identifiers.copy(), used_replacements.copy()))

        prompts, generations, results, mean_logprobs = target.generate_test(transformed_examples)
        for tr_example, prompt, generation, result, mean_logprobs in zip(transformed_examples, prompts, generations, results, mean_logprobs):
            generation = generation[0] # pass 1
            result = result[0] # pass 1
            mean_logprob = mean_logprobs[0] # pass 1
            pbar.write(f"{str([i.name.decode() for i in tr_example.identifiers])} -> {str([r for r in tr_example.replacements])}: {result}: perplexity={perplexity(mean_logprob):.4f}")
            attempts += 1
            if not result.passed and not result.timed_out:
                pbar.write(f"Successfully attacked example after {attempts} attempts, replaced {len(tr_example.identifiers)} identifiers.")
                pbar.write(str(prompt))
                pbar.write("-"*80)
                pbar.write(generation)
                return Result.from_success(example, attempts, target.queries, generation, result.share_tests_passed, tr_example, time_start)

        transformed_examples = []

    pbar.write(f"Failed to attack example after {target.queries} attempts.")
    return Result.from_failure(example, attempts, target.queries, time_start)


def parse_args():
    parser = common.ArgumentParser()
    parser.add_argument("--from_scratch", action="store_true", help="Rewrite the cache")
    parser.add_argument("--allow_multi_gpu", action="store_true", help="Allow multi-gpu training when available")
    parser.add_argument("--max_workers", type=int, default=multiprocessing.cpu_count(), help="Maximum number of workers to use")
    parser.add_argument("--use_logprobs", action="store_true", help="Use logprobs for the attack")
    return parser.parse_args()


def attack(args: argparse.Namespace):
    if args.use_logprobs and "sonnet" in args.model_name:
        raise ValueError("Logprobs are not supported for Sonnet models")
    print("Loading target model...")
    with TestRunnerManager(args.max_workers) as tmgr:
        if args.model_name in common.CODE_GENERATION_API_MODELS:
            target = APIModel(args.model_name, tmgr, args.max_workers)
        else:
            target = Model(args.model_name, tmgr, args.allow_multi_gpu)
        print("Preparing attack...")
        repl_ids = preprocess(args)
        test_examples = load_examples(args.dataset)
        print("Filtering examples...")
        filtered = get_filtered_examples(test_examples, target)
        print(f"Filtered {len(filtered)}/{len(test_examples)} examples that the target model got right")

        print("Loading successful replacement pairs...")
        with open("task_to_ids_replacements.json") as f:
            repl_pairs_by_taskid = json.load(f)

        print(f"Setting seed to {args.seed}")
        random.seed(args.seed)

        print("Starting attack...")

        stats = AttackStats()
        with open(args.results_path, "w") as f:
            for i, example in enumerate(pbar := tqdm(filtered), 1):
                pbar.write(f"Attacking example {i}: {example.task['task_id']} / {example.task['entry_point']}")
                repl_pairs = repl_pairs_by_taskid[example.task["task_id"]]
                result = attack_single_combs_biased(target, example, repl_ids, repl_pairs, pbar)
                stats.update(result)
                pbar.set_description(str(stats))
                f.write(json.dumps(result._asdict()) + "\n")
        print_results(args.results_path)
