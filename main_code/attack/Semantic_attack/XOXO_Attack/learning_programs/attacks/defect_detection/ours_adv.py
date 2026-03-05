import argparse
import json
import multiprocessing
import os
import random
import time
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import learning_programs.attacks.defect_detection.ours
from learning_programs.attacks import common
from learning_programs.attacks.defect_detection.attack_utils import (
    Model,
    initialize_target_model
)
from learning_programs.datasets.defect_detection import load_examples, Example
from learning_programs.transforms.processor import apply_transforms
from learning_programs.transforms.c.processor import C_PARSER
from learning_programs.transforms.tree_utils import get_nodes
from learning_programs.attacks.defect_detection.ours import (
    preprocess,
    filter_repl_ids,
    find_identifiers_in_trunc_code,
    Result,
    _tokenize,
    BATCH_SIZES,
    get_filtered_examples,
    AttackStats
)

def attack_single(target: Model, tokenizer: AutoTokenizer, example: Example, repl_ids: list[str], args: argparse.Namespace, pbar: tqdm, train_attack: bool, quiet: bool = True) -> tuple[Result, str | None]:
    time_start = time.time()
    # Since this is the training set, we must check for parsing errors
    tree = C_PARSER.parse(bytes(example.code, "utf8"))
    if any(node.is_error or node.is_missing for node in get_nodes(tree)):
        if not quiet:
            pbar.write("Skipping example with parse errors")
        return Result.from_parser_failure(example, time_start), None

    identifiers = find_identifiers_in_trunc_code(example.code, tokenizer, args.block_size)
    if len(identifiers) == 0:
        if not quiet:
            pbar.write("Skipping example with no identifiers")
        return Result.from_parser_failure(example, time_start), None
    num_replacement_candidates = 300 if len(identifiers) > 1 else 1500

    attempts = 0
    orig_prob = target(_tokenize(example.code, tokenizer, args)).item()
    best_prob = orig_prob
    best_prob_by_id = {i: orig_prob for i in identifiers}
    best_repl_by_id = defaultdict(list)
    best_attempt = 0
    best_new_code = None
    found_new_best = False
    tried_code = dict()

    repl_ids = filter_repl_ids(repl_ids, identifiers)
    replacement_candidates = repl_ids[:num_replacement_candidates] if train_attack else random.sample(repl_ids, k=num_replacement_candidates)
    for round, replacement in enumerate(replacement_candidates, 1):
        for identifier in identifiers:
            new_code = apply_transforms(example.code, identifier.to_transforms(replacement.encode()))
            prob = target(_tokenize(new_code, tokenizer, args)).item()
            pred = prob > 0.5
            attempts += 1
            if pred != bool(example.label):
                if not quiet:
                    pbar.write(f"Successfully attacked example after {attempts} attempts, replaced 1 identifier: {identifier.name.decode()} -> {replacement}")
                return Result.from_success(example, attempts, new_code, 1, len(identifier.locations), time_start), new_code
            if prob == (best_prob := min(best_prob, prob) if example.label else max(best_prob, prob)):
                best_attempt = attempts
                best_new_code = new_code
            if example.label and prob < best_prob_by_id[identifier] or (not example.label and prob > best_prob_by_id[identifier]):
                found_new_best = True
                best_prob_by_id[identifier] = prob
                best_repl_by_id[identifier].append((replacement, prob))
                if not quiet:
                    pbar.write(f"Best prob for {identifier.name.decode()}: {replacement}: {orig_prob:.4f} -> {best_prob_by_id[identifier]:.4f}")

        if round % 10 == 0 and found_new_best and sum(prob != orig_prob for prob in best_prob_by_id.values()) > 1 and len(identifiers) > 1:
            if not quiet:
                pbar.write(f"Round {round}: Attempted {attempts} replacements, best prob: {best_prob:.4f} (after {best_attempt}/{attempts} attempts)")
            best_prob = orig_prob
            best_prob_id_repl = sorted([(i, r, p) for i in identifiers for (r, p) in best_repl_by_id[i]], key=lambda x: x[2], reverse=not example.label)
            used_transforms = []
            used_identifiers = set()
            used_replacements = set()
            for identifier, replacement, _ in best_prob_id_repl:
                if identifier in used_identifiers or replacement in used_replacements:
                    continue
                new_transforms = used_transforms + identifier.to_transforms(replacement.encode())
                new_code = apply_transforms(example.code, new_transforms)
                if new_code not in tried_code:
                    tried_code[new_code] = target(_tokenize(new_code, tokenizer, args)).item()
                    attempts += 1
                prob = tried_code[new_code]
                pred = prob > 0.5
                if (example.label and prob <= best_prob) or (not example.label and prob >= best_prob):
                    used_identifiers.add(identifier)
                    used_replacements.add(replacement)
                    used_transforms = new_transforms
                    if not quiet:
                        pbar.write(f"Combining {identifier.name.decode()} -> {replacement} with existing transforms improved prob: {best_prob:.4f} -> {prob:.4f}")
                else:
                    if not quiet:
                        pbar.write(f"Combining {identifier.name.decode()} -> {replacement} with existing transforms did not improve prob: {best_prob:.4f} -> {prob:.4f}")
                if pred != bool(example.label):
                    if not quiet:
                        pbar.write(f"Successfully attacked example after {attempts} attempts, replaced {len(used_identifiers)} identifiers: {', '.join(i.name.decode() for i in used_identifiers)}")
                    return Result.from_success(example, attempts, new_code, len(used_identifiers), sum(len(i.locations) for i in used_identifiers), time_start), new_code
                if prob == (best_prob := min(best_prob, prob) if example.label else max(best_prob, prob)):
                    best_attempt = attempts
                    best_new_code = new_code
            found_new_best = False

    if not quiet:
        pbar.write(
            f"Failed to attack example: orig prob: {orig_prob:.4f}, best prob: {best_prob:.4f} (after {best_attempt}/{attempts} attempts), num_identifiers: {len(identifiers)}, num_attempts: {attempts}"
        )
    return Result.from_failure(example, attempts, time_start), best_new_code


MAX_PARTS = 8


def parse_args():
    parser = common.ArgumentParser()
    parser.add_argument("--from_scratch", action="store_true", help="Rewrite cached data")
    parser.add_argument("--part", type=int, required=True, help=f"Part of the training data to use (0-{MAX_PARTS - 1})")
    parser.add_argument("--split", type=str, required=True, help="Dataset split to attack", choices=["train", "valid"])
    args = parser.parse_args()
    args.allow_multi_gpu = False
    args.batch_size = BATCH_SIZES[args.model_name]
    return args


def par_gcb_attack(args: argparse.Namespace):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        learning_programs.attacks.defect_detection.ours.MP_POOL = pool
        attack(args)


def attack(args: argparse.Namespace):
    target, tokenizer = initialize_target_model(args)
    print("Preparing attack...")
    repl_ids = preprocess(args)
    print("Starting attack...")
    test_examples = load_examples(args.split)
    test_examples = test_examples[args.part * len(test_examples) // MAX_PARTS: (args.part + 1) * len(test_examples) // MAX_PARTS]
    print("Filtering examples...")
    filtered = get_filtered_examples(test_examples, target, tokenizer, args)
    print(f"Filtered {len(filtered)}/{len(test_examples)} examples that the target model got right")

    random.seed(args.seed)

    stats = AttackStats()
    results_path = os.path.join(args.results_dir, f"results_{args.split}_{args.part}.jsonl")
    with open(results_path, "w") as f:
        for i, example in enumerate(pbar := tqdm(filtered), 1):
            pbar.write(f"Attacking example {i} (label={example.label})")
            result, adv_ex_code = attack_single(target, tokenizer, example, repl_ids, args, pbar, False, False)
            stats.update(result)
            pbar.set_description(str(stats))
            if adv_ex_code is not None:
                adv_ex = Example(example.idx, adv_ex_code, example.label)
                f.write(json.dumps(adv_ex._asdict()) + "\n")


if __name__ == "__main__":
    with torch.no_grad():
        args = parse_args()
        attack_fn = par_gcb_attack if args.model_name == "microsoft/graphcodebert-base" else attack
        attack_fn(args)

