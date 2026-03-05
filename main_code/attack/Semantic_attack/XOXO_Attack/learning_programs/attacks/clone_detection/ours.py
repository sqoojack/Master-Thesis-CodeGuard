import argparse
import json
import multiprocessing
import os
import random
import time
from collections import defaultdict
from typing import NamedTuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from learning_programs.attacks import common
from learning_programs.attacks.clone_detection.attack_utils import (
    convert_code_to_features_gcb,
    convert_code_to_features_raw_gcb,
    gcb_attn_mask,
    gcb_attn_mask_raw,
    Model,
    initialize_target_model
)
from learning_programs.attacks.print_results import print_results, Result as BaseResult
from learning_programs.datasets.clone_detection import load_examples, Example
from learning_programs.transforms.processor import apply_transforms
from learning_programs.transforms.transform import Identifier
from learning_programs.transforms.java.processor import Processor


class Result(BaseResult):
    @classmethod
    def from_failure(cls, example: Example, queries: int, time_start: float) -> "Result":
        return cls(example.idx, False, queries, example.code1 + example.code2, None, example.label, None, None, time.time() - time_start)

    @classmethod
    def from_success(cls, example: Example, queries: int, new_code: str, num_changed_vars: int, num_changed_pos: int, time_start: float) -> "Result":
        return cls(example.idx, True, queries, example.code1 + example.code2, new_code, example.label, num_changed_vars, num_changed_pos, time.time() - time_start)

    @classmethod
    def from_parser_failure(cls, example: Example, time_start: float) -> "Result":
        return cls(example.idx, None, None, example.code1 + example.code2, None, example.label, None, None, time.time() - time_start)


def _tqdm_batch(iterable, n=1):
    len_ = len(iterable)
    for ndx in tqdm(range(0, len_, n)):
        yield iterable[ndx : min(ndx + n, len_)]


def _batch(iterable, n=1):
    len_ = len(iterable)
    for ndx in range(0, len_, n):
        yield iterable[ndx : min(ndx + n, len_)]


def get_unique_id_names(split: str) -> list[str]:
    codes = set()
    for example in load_examples(split) + load_examples(f"{split}_ours"):
        codes.add(example.code1)
        codes.add(example.code2)
    processor = Processor()
    id_names = set()
    for code in tqdm(codes):
        id_names |= {i.name.decode() for i in processor.find_identifiers(code)}
    return sorted(id_names)  # Sort for reproducibility


def preprocess(args: argparse.Namespace) -> list[str]:
    save_path_ids = os.path.join(args.results_dir, "ids.json")

    if not args.from_scratch and os.path.exists(save_path_ids):
        print(f"Found existing data at {args.results_dir}, loading...")
        with open(save_path_ids) as f:
            return json.load(f)

    print("Scraping training data for identifiers...")
    id_names = get_unique_id_names("train")
    with open(save_path_ids, "w") as f:
        json.dump(id_names, f)
    print(f"Saved unique identifier names to {save_path_ids}")
    return id_names


class TransformedCode(NamedTuple):
    orig_prob: float
    replacement: str
    code1: str
    code2: str


def compute_scores(totals: dict[str, float], counts: dict[str, int], repl_ids: list[str]) -> list[float]:
    return [totals[repl_id] / counts[repl_id] if repl_id in totals else 0.0 for repl_id in repl_ids]


def convert_code_to_features_worker(inputs):
    code1, code2, tokenizer, args = inputs
    return convert_code_to_features_gcb(code1, code2, tokenizer, None, args)


def _tokenize_batch(batch: list[TransformedCode | Example], tokenizer: AutoTokenizer, args: argparse.Namespace) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if args.model_name == "microsoft/graphcodebert-base":
        inpfs = MP_POOL.map(convert_code_to_features_worker, [(e.code1, e.code2, tokenizer, args) for e in batch])
        input_ids_1 = torch.tensor([inp.input_ids_1 for inp in inpfs], dtype=torch.long).to(args.device)
        position_idx_1 = torch.tensor([inp.position_idx_1 for inp in inpfs], dtype=torch.long).to(args.device)
        input_ids_2 = torch.tensor([inp.input_ids_2 for inp in inpfs], dtype=torch.long).to(args.device)
        position_idx_2 = torch.tensor([inp.position_idx_2 for inp in inpfs], dtype=torch.long).to(args.device)
        attn_mask_1,  attn_mask_2 = list(zip(*[gcb_attn_mask(inpf, args) for inpf in inpfs]))
        attn_mask_1 = torch.stack(attn_mask_1).to(args.device)
        attn_mask_2 = torch.stack(attn_mask_2).to(args.device)
        return (input_ids_1, position_idx_1, attn_mask_1, input_ids_2, position_idx_2, attn_mask_2)
    tokenized1 = tokenizer([e.code1 for e in batch], padding="max_length", truncation=True, return_tensors="pt", max_length=args.block_size).input_ids.to(args.device)
    tokenized2 = tokenizer([e.code2 for e in batch], padding="max_length", truncation=True, return_tensors="pt", max_length=args.block_size).input_ids.to(args.device)
    return torch.cat((tokenized1, tokenized2), dim=1)


def _tokenize(code: str, tokenizer: AutoTokenizer, args: argparse.Namespace) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if args.model_name == "microsoft/graphcodebert-base":
        _, input_ids, position_idx, dfg_to_code, dfg_to_dfg = convert_code_to_features_raw_gcb(code, tokenizer, args)
        attn_mask = gcb_attn_mask_raw(input_ids, position_idx, dfg_to_code, dfg_to_dfg, args).unsqueeze(0).to(args.device)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(args.device)
        position_idx = torch.tensor(position_idx, dtype=torch.long).unsqueeze(0).to(args.device)
        return input_ids, position_idx, attn_mask
    return tokenizer(code, padding="max_length", truncation=True, return_tensors="pt", max_length=args.block_size).input_ids.to(args.device)


def _tokenize_first(code1: str, tokenized: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor], tokenizer: AutoTokenizer, args: argparse.Namespace) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokenized1 = _tokenize(code1, tokenizer, args)
    return _join_first(tokenized1, tokenized, args)


def _join_first(tokenized1: torch.Tensor | tuple[torch.Tensor,...], tokenized: torch.Tensor | tuple[torch.Tensor,...], args: argparse.Namespace) -> torch.Tensor | tuple[torch.Tensor,...]:
    if args.model_name == "microsoft/graphcodebert-base":
        return tokenized1[0], tokenized1[1], tokenized1[2], tokenized[0], tokenized[1], tokenized[2]
    return torch.cat((tokenized1, tokenized), dim=1)


def _tokenize_second(code2: str, tokenized: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor], tokenizer: AutoTokenizer, args: argparse.Namespace) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokenized2 = _tokenize(code2, tokenizer, args)
    return _join_second(tokenized2, tokenized, args)


def _join_second(tokenized2: torch.Tensor | tuple[torch.Tensor,...], tokenized: torch.Tensor | tuple[torch.Tensor,...], args: argparse.Namespace) -> torch.Tensor | tuple[torch.Tensor,...]:
    if args.model_name == "microsoft/graphcodebert-base":
        return tokenized2[0], tokenized2[1], tokenized2[2], tokenized[0], tokenized[1], tokenized[2]
    return torch.cat((tokenized, tokenized2), dim=1)


def train(repl_ids: list[str], surrogate: Model, tokenizer: AutoTokenizer, args: argparse.Namespace) -> list[float]:
    time_start = time.time()
    save_path = os.path.join(args.results_dir, "scores.json")

    if not args.from_scratch and os.path.exists(save_path):
        print(f"Found existing scores at {save_path}, loading...")
        with open(save_path) as f:
            return json.load(f)

    random.seed(args.seed)
    print("Filtering train and validation examples...")
    train_examples = load_examples("train_ours")
    train_filtered = get_filtered_examples(train_examples, surrogate, tokenizer, args)
    print(f"Filtered {len(train_filtered)}/{len(train_examples)} training examples that the target model got right")
    valid_examples = load_examples("valid_ours")
    valid_filtered = get_filtered_examples(valid_examples, surrogate, tokenizer, args)
    print(f"Filtered {len(valid_filtered)}/{len(valid_examples)} validation examples that the target model got right")
    counts = defaultdict(int)
    totals = defaultdict(float)
    training_attempts = 0
    num_epochs = 10
    num_replacements_per_epoch = 5
    best_valid_rate = 0.0
    best_attempt_rate = float("inf")

    print("Getting original probabilities...")
    orig_probs_map = []
    for batch in tqdm(_batch(train_filtered, n=args.batch_size), total=len(train_filtered) // args.batch_size):
        probs = surrogate(_tokenize_batch(batch, tokenizer, args))
        orig_probs_map.extend(probs.tolist())
        training_attempts += len(batch)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")

        print("Generating transformed code...")
        transformed = []
        for i, example in enumerate(pbar := tqdm(train_filtered)):
            identifiers1 = find_identifiers_in_trunc_code(example.code1, tokenizer, args.block_size)
            identifiers2 = find_identifiers_in_trunc_code(example.code2, tokenizer, args.block_size)
            if len(identifiers1) + len(identifiers2) == 0:
                continue
            for identifier in identifiers1:
                for replacement in random.sample(filter_repl_ids(repl_ids, identifiers1), num_replacements_per_epoch):
                    new_code = apply_transforms(example.code1, identifier.to_transforms(replacement.encode()))
                    transformed.append(TransformedCode(orig_probs_map[i], replacement, new_code, example.code2))
            for identifier in identifiers2:
                for replacement in random.sample(filter_repl_ids(repl_ids, identifiers2), num_replacements_per_epoch):
                    new_code = apply_transforms(example.code2, identifier.to_transforms(replacement.encode()))
                    transformed.append(TransformedCode(orig_probs_map[i], replacement, example.code1, new_code))

        print("Collecting model feedback...")
        for i, batch in enumerate(pbar := tqdm(_batch(transformed, args.batch_size), total=len(transformed) // args.batch_size)):
            orig_probs = torch.tensor([e.orig_prob for e in batch], dtype=torch.float).to(args.device)
            probs = surrogate(_tokenize_batch(batch, tokenizer, args))
            training_attempts += len(batch)
            diffs = probs - orig_probs
            diffs[orig_probs > 0.5] *= -1
            for example, diff in zip(batch, diffs):
                counts[example.replacement] += 1
                totals[example.replacement] += diff.item()

            # Show top 5 best performing identifiers every 100 batches
            if i % 100 == 0 and i > 0:
                pbar.write(f"Top 5 best performing identifiers after {i*args.batch_size} transforms:")
                for rank, k in enumerate(sorted(totals, key=lambda k: totals[k] / counts[k], reverse=True)[:5], 1):
                    pbar.write(f"{rank}: {k}: {totals[k] / counts[k]:.4f} ({counts[k]})")

        print("Validating...")
        scores = compute_scores(totals, counts, repl_ids)
        ranked_repl_ids = [repl_ids[i] for i in np.argsort(scores)[::-1]]
        stats = AttackStats()
        for example in (pbar := tqdm(valid_filtered)):
            result = attack_single(surrogate, tokenizer, example, ranked_repl_ids, args, pbar, True)
            training_attempts += result.queries
            stats.update(result)
            pbar.set_description(str(stats))

        if stats.success_rate_valid > best_valid_rate or (stats.success_rate_valid == best_valid_rate and stats.attempt_rate < best_attempt_rate):
            print(f"New best success rate: {stats.success_rate_valid:.2%} (was {best_valid_rate:.2%}), saving checkpoint")
            best_valid_rate = stats.success_rate_valid
            best_attempt_rate = stats.attempt_rate
            with open(save_path, "w") as f:
                print(f"Saving scores to {save_path}")
                json.dump(scores, f)
        else:
            print(f"Success rate: {stats.success_rate_valid:.2%} (best: {best_valid_rate:.2%}), early stopping")
            break

    time_elapsed = time.time() - time_start
    training_log_path = os.path.join(args.results_dir, "training_log.json")
    with open(training_log_path, "w") as f:
        print(f"Saving training log to {training_log_path}")
        json.dump(
            {
                "finished_training_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "training_attempts": training_attempts,
                "time_elapsed": time_elapsed,
                "best_valid_rate": best_valid_rate,
                "best_attempt_rate": best_attempt_rate,
                "num_epochs": epoch + 1,
            },
            f,
        )

    return scores


def get_filtered_examples(examples: list[Example], target: Model, tokenizer: AutoTokenizer, args: argparse.Namespace) -> list[Example]:
    filtered = []
    for batch in _batch(examples, args.batch_size):
        labels = torch.tensor([example.label for example in batch], dtype=torch.bool).to(args.device)
        probs = target(_tokenize_batch(batch, tokenizer, args))
        preds = (probs > 0.5).squeeze()
        filtered.extend([batch[i] for i, eq in enumerate(preds == labels) if eq])
    return filtered


def test_model(args: argparse.Namespace):
    model, tokenizer = initialize_target_model(args)
    examples = load_examples("test")
    all_labels = []
    all_preds = []
    args.allow_multi_gpu = True
    with MultiGPUContext(model, args) as ctx:
        for batch in _tqdm_batch(examples, ctx.args.batch_size):
            labels = torch.tensor([example.label for example in batch], dtype=torch.bool).to(ctx.args.device)
            probs = ctx.model(_tokenize_batch(batch, tokenizer, ctx.args)).squeeze()
            preds = (probs > 0.5).squeeze()
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    acc = (all_labels == all_preds).float().mean().item()
    tp = (all_labels & all_preds).float().sum()
    rec = (tp / all_labels.float().sum()).item()
    prec = (tp / all_preds.float().sum()).item()
    f1 = 2 * prec * rec / (prec + rec)
    print(f"Accuracy: {acc:.2%}")
    print(f"Precision: {prec:.2%}")
    print(f"Recall: {rec:.2%}")
    print(f"F1: {f1:.2%}")
    output_path = os.path.join(args.target_weights_dir, "test_results.json")
    print(f"Saving performance to {output_path}")
    with open(output_path, "w") as f:
        json.dump({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}, f)


class MultiGPUContext:
    model: Model
    args: argparse.Namespace

    def __init__(self, model: Model, args: argparse.Namespace):
        self.model = model
        self.args = args

    def __enter__(self):
        self.prev_model = self.model
        self.prev_batch_size = self.args.batch_size
        self.prev_device = self.args.device

        if self.args.allow_multi_gpu and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.args.batch_size *= torch.cuda.device_count()
            self.args.device = torch.device("cuda")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model = self.prev_model
        self.args.batch_size = self.prev_batch_size
        self.args.device = self.prev_device


def truncate_code_to_block_size(code: str, tokenizer: AutoTokenizer, block_size: int) -> str:
    truncated_tokens = tokenizer.tokenize(code, truncation=True, max_length=block_size)[:block_size]
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(truncated_tokens))


def find_identifiers_in_trunc_code(code: str, tokenizer: AutoTokenizer, block_size: int) -> list[Identifier]:
    processor = Processor()
    trunc_code_len = len(bytes(truncate_code_to_block_size(code, tokenizer, block_size), "utf-8"))
    ids = processor.find_identifiers(code)
    return [i for i in ids if any(loc.end_byte < trunc_code_len for loc in i.locations)]


def filter_repl_ids(repl_ids: list[str], identifiers: list[Identifier]) -> list[str]:
    id_names = {i.name.decode() for i in identifiers}
    return [repl_id for repl_id in repl_ids if repl_id not in id_names]


def attack_single(target: Model, tokenizer: AutoTokenizer, example: Example, repl_ids: list[str], args: argparse.Namespace, pbar: tqdm, train_attack: bool, quiet: bool = True) -> Result:
    time_start = time.time()
    identifiers1 = find_identifiers_in_trunc_code(example.code1, tokenizer, args.block_size)
    identifiers2 = find_identifiers_in_trunc_code(example.code2, tokenizer, args.block_size)
    if len(identifiers1) == 0 and len(identifiers2) == 0:
        if not quiet:
            pbar.write("Skipping example with no identifiers")
        return Result.from_parser_failure(example, time_start)
    num_replacement_candidates = 300 if len(identifiers1) + len(identifiers2) > 2 else 1500
    max_attempts = 2000

    attempts = 0
    tokenized1 = _tokenize(example.code1, tokenizer, args)
    tokenized2 = _tokenize(example.code2, tokenizer, args)
    input_ids = _join_first(tokenized1, tokenized2, args)
    orig_prob = target(input_ids).item()
    best_prob = orig_prob
    best_prob_by_id1 = {i: orig_prob for i in identifiers1}
    best_prob_by_id2 = {i: orig_prob for i in identifiers2}
    best_repl_by_id1 = defaultdict(list)
    best_repl_by_id2 = defaultdict(list)
    best_attempt = 0
    found_new_best = False
    tried_code = dict()

    repl_ids_1 = filter_repl_ids(repl_ids, identifiers1)
    repl_ids_2 = filter_repl_ids(repl_ids, identifiers2)
    replacement_candidates1 = repl_ids_1[:num_replacement_candidates] if train_attack else random.sample(repl_ids_1, k=num_replacement_candidates)
    replacement_candidates2 = repl_ids_2[:num_replacement_candidates] if train_attack else random.sample(repl_ids_2, k=num_replacement_candidates)
    for round, (replacement1, replacement2) in enumerate(zip(replacement_candidates1, replacement_candidates2), 1):
        for identifier in identifiers1:
            new_code = apply_transforms(example.code1, identifier.to_transforms(replacement1.encode()))
            prob = target(_tokenize_first(new_code, tokenized2, tokenizer, args)).item()
            pred = prob > 0.5
            attempts += 1
            if pred != bool(example.label):
                if not quiet:
                    pbar.write(f"Successfully attacked example after {attempts} attempts, replaced 1 identifier: {identifier.name.decode()} -> {replacement1}")
                return Result.from_success(example, attempts, new_code + example.code2, 1, len(identifier.locations), time_start)
            if prob == (best_prob := min(best_prob, prob) if example.label else max(best_prob, prob)):
                best_attempt = attempts
            if example.label and prob < best_prob_by_id1[identifier] or (not example.label and prob > best_prob_by_id1[identifier]):
                found_new_best = True
                best_prob_by_id1[identifier] = prob
                best_repl_by_id1[identifier].append((replacement1, prob))
                if not quiet:
                    pbar.write(f"Best prob for {identifier.name.decode()}: {replacement1}: {orig_prob:.4f} -> {best_prob_by_id1[identifier]:.4f}")
            if attempts >= max_attempts:
                break
        
        if attempts >= max_attempts:
            break

        for identifier in identifiers2:
            new_code = apply_transforms(example.code2, identifier.to_transforms(replacement2.encode()))
            prob = target(_tokenize_second(new_code, tokenized1, tokenizer, args)).item()
            pred = prob > 0.5
            attempts += 1
            if pred != bool(example.label):
                if not quiet:
                    pbar.write(f"Successfully attacked example after {attempts} attempts, replaced 1 identifier: {identifier.name.decode()} -> {replacement2}")
                return Result.from_success(example, attempts, example.code1 + new_code, 1, len(identifier.locations), time_start)
            if prob == (best_prob := min(best_prob, prob) if example.label else max(best_prob, prob)):
                best_attempt = attempts
            if example.label and prob < best_prob_by_id2[identifier] or (not example.label and prob > best_prob_by_id2[identifier]):
                found_new_best = True
                best_prob_by_id2[identifier] = prob
                best_repl_by_id2[identifier].append((replacement2, prob))
                if not quiet:
                    pbar.write(f"Best prob for {identifier.name.decode()}: {replacement2}: {orig_prob:.4f} -> {best_prob_by_id2[identifier]:.4f}")
            if attempts >= max_attempts:
                break

        if attempts >= max_attempts:
            break

        if round % 10 == 0 and found_new_best:
            if not quiet:
                pbar.write(f"Round {round}: Attempted {attempts} replacements, best prob: {best_prob:.4f} (after {best_attempt}/{attempts} attempts)")
            best_prob = orig_prob

            best_prob_id_repl_code = [(i, r, p, True) for i in identifiers1 for (r, p) in best_repl_by_id1[i]] + [(i, r, p, False) for i in identifiers2 for (r, p) in best_repl_by_id2[i]]
            best_prob_id_repl_code.sort(key=lambda x: x[2], reverse=not example.label)
            used_transforms1 = []
            used_transforms2 = []
            used_tokenized1 = tokenized1
            used_tokenized2 = tokenized2
            used_identifiers1 = set()
            used_identifiers2 = set()
            used_replacements1 = set()
            used_replacements2 = set()
            for identifier, replacement, _, is_code1 in best_prob_id_repl_code:
                if is_code1:
                    if identifier in used_identifiers1 or replacement in used_replacements1:
                        continue
                    new_transforms = used_transforms1 + identifier.to_transforms(replacement.encode())
                    new_code = apply_transforms(example.code1, new_transforms)
                    if new_code not in tried_code:
                        tokenized = _tokenize(new_code, tokenizer, args)
                        tried_code[new_code] = target(_join_first(tokenized, used_tokenized2, args)).item()
                        attempts += 1
                    prob = tried_code[new_code]
                    pred = prob > 0.5
                    if (example.label and prob <= best_prob) or (not example.label and prob >= best_prob):
                        used_code = new_code + example.code2
                        used_identifiers1.add(identifier)
                        used_replacements1.add(replacement)
                        used_tokenized1 = tokenized
                        used_transforms1 = new_transforms
                        if not quiet:
                            pbar.write(f"Combining {identifier.name.decode()} -> {replacement} with existing transforms improved prob: {best_prob:.4f} -> {prob:.4f}")
                    else:
                        if not quiet:
                            pbar.write(f"Combining {identifier.name.decode()} -> {replacement} with existing transforms did not improve prob: {best_prob:.4f} -> {prob:.4f}")
                else:
                    if identifier in used_identifiers2 or replacement in used_replacements2:
                        continue
                    new_transforms = used_transforms2 + identifier.to_transforms(replacement.encode())
                    new_code = apply_transforms(example.code2, new_transforms)
                    if new_code not in tried_code:
                        tokenized = _tokenize(new_code, tokenizer, args)
                        tried_code[new_code] = target(_join_second(tokenized, used_tokenized1, args)).item()
                        attempts += 1
                    prob = tried_code[new_code]
                    pred = prob > 0.5
                    if (example.label and prob <= best_prob) or (not example.label and prob >= best_prob):
                        used_code = example.code1 + new_code
                        used_identifiers2.add(identifier)
                        used_replacements2.add(replacement)
                        used_tokenized2 = tokenized
                        used_transforms2 = new_transforms
                        if not quiet:
                            pbar.write(f"Combining {identifier.name.decode()} -> {replacement} with existing transforms improved prob: {best_prob:.4f} -> {prob:.4f}")
                    else:
                        if not quiet:
                            pbar.write(f"Combining {identifier.name.decode()} -> {replacement} with existing transforms did not improve prob: {best_prob:.4f} -> {prob:.4f}")
                if pred != bool(example.label):
                    num_used_identifiers = len(used_identifiers1) + len(used_identifiers2)
                    num_replaced_locs = sum(len(i.locations) for i in used_identifiers1) + sum(len(i.locations) for i in used_identifiers2)
                    if not quiet:
                        pbar.write(
                            f"Successfully attacked example after {attempts} attempts, replaced {num_used_identifiers} identifiers: 1: {', '.join(i.name.decode() for i in used_identifiers1)} 2: {', '.join(i.name.decode() for i in used_identifiers2)}"
                        )
                    return Result.from_success(example, attempts, used_code, num_used_identifiers, num_replaced_locs, time_start)
                if prob == (best_prob := min(best_prob, prob) if example.label else max(best_prob, prob)):
                    best_attempt = attempts
                if attempts >= max_attempts:
                    break
            found_new_best = False
            if attempts >= max_attempts:
                break
            

    if not quiet:
        pbar.write(
            f"Failed to attack example: orig prob: {orig_prob:.4f}, best prob: {best_prob:.4f} (after {best_attempt}/{attempts} attempts), num_identifiers: {len(identifiers1) + len(identifiers2)}, num_attempts: {attempts}"
        )
    return Result.from_failure(example, attempts, time_start)


class AttackStats:
    def __init__(self):
        self.successes = 0
        self.attempts = 0
        self.ids_replaced = 0
        self.pos_replaced = 0
        self.valid_examples = 0
        self.total_examples = 0

    def update(self, result: Result):
        self.total_examples += 1
        if result.success is not None:
            self.valid_examples += 1
            self.attempts += result.queries
        if result.success:
            self.successes += 1
            self.ids_replaced += result.num_changed_vars
            self.pos_replaced += result.num_changed_pos
        self.success_rate = self.successes / self.total_examples
        self.success_rate_valid = self.successes / self.valid_examples if self.valid_examples else 0
        self.attempt_rate = self.attempts / self.valid_examples if self.valid_examples else 0
        self.ids_replacement_rate = self.ids_replaced / self.successes if self.successes else 0
        self.pos_replacement_rate = self.pos_replaced / self.successes if self.successes else 0

    def __str__(self):
        return f"SRV: {self.success_rate_valid:.2%}, " f"SR: {self.success_rate:.2%}, " f"AR: {self.attempt_rate:.2f}, " f"IR: {self.ids_replacement_rate:.2f}, " f"PR: {self.pos_replacement_rate:.2f}"


BATCH_SIZES = {
    "microsoft/codebert-base": 128,
    "microsoft/graphcodebert-base": 128,
    "Salesforce/codet5p-110m-embedding": 64,
    "Salesforce/codet5p-220m": 64,
}


def parse_args():
    parser = common.ArgumentParser()
    parser.add_argument("--from_scratch", action="store_true", help="Rewrite the cache")
    parser.add_argument("--allow_multi_gpu", action="store_true", help="Allow multi-gpu training when available")
    args = parser.parse_args()
    args.batch_size = BATCH_SIZES[args.model_name]
    return args


def attack(args: argparse.Namespace, train_attack: bool):
    target, tokenizer = initialize_target_model(args)

    print("Preparing attack...")
    repl_ids = preprocess(args)
    if train_attack:
        print("Training attacker...")
        with MultiGPUContext(target, args) as ctx:
            scores = train(repl_ids, ctx.model, tokenizer, ctx.args)
    print("Starting attack...")
    test_examples = load_examples("test")
    print("Filtering examples...")
    with MultiGPUContext(target, args) as ctx:
        filtered = get_filtered_examples(test_examples, ctx.model, tokenizer, ctx.args)
    print(f"Filtered {len(filtered)}/{len(test_examples)} examples that the target model got right")

    if train_attack:
        # Sort repl_ids by scores
        repl_ids = [repl_ids[i] for i in np.argsort(scores)[::-1]]
    else:
        random.seed(args.seed)

    stats = AttackStats()
    with open(args.results_path, "w") as f:
        for i, example in enumerate(pbar := tqdm(filtered), 1):
            pbar.write(f"Attacking example {i} (label={example.label})")
            result = attack_single(target, tokenizer, example, repl_ids, args, pbar, train_attack, False)
            stats.update(result)
            pbar.set_description(str(stats))
            f.write(json.dumps(result._asdict()) + "\n")
    print_results(args.results_path)


MP_POOL = None


def par_gcb_attack(args: argparse.Namespace, train_attack: bool):
    global MP_POOL
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        MP_POOL = pool
        attack(args, train_attack)


if __name__ == "__main__":
    with torch.no_grad():
        args = parse_args()
        attack_fn = par_gcb_attack if args.model_name == "microsoft/graphcodebert-base" else attack
        attack_fn(args, train_attack=False)
