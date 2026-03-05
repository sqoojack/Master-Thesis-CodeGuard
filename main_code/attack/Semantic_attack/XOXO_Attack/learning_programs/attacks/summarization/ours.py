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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from learning_programs.attacks import common
from learning_programs.attacks.summarization.attack_utils import initialize_target_model, Seq2SeqCodeBERT, get_criterion, BLEU_ZERO_ATOL
from learning_programs.attacks.print_results import print_results, SummarizationResult as BaseResult
from learning_programs.datasets.summarization import load_examples, Example
from learning_programs.metrics.bleu import compute_bleu, compute_bleus
from learning_programs.transforms.processor import apply_transforms
from learning_programs.transforms.transform import Identifier
from learning_programs.transforms.python.processor import Processor
from learning_programs.attacks.defect_detection.ours import AttackStats as BaseAttackStats


class Result(BaseResult):
    @classmethod
    def from_failure(cls, example: Example, queries: int, orig_bleu: float, orig_docstring: str, best_bleu: float, time_start: float) -> "Result":
        return cls(example.idx, False, queries, orig_bleu, orig_docstring, example.code, example.docstring, None, None, best_bleu, None, None, time.time() - time_start)

    @classmethod
    def from_success(cls, example: Example, queries: int, orig_bleu: float, orig_docstring: str, new_code: str, new_docstring: str, best_bleu: float, num_changed_vars: int, num_changed_pos: int, time_start: float) -> "Result":
        return cls(example.idx, True, queries, orig_bleu, orig_docstring, example.code, example.docstring, new_code, new_docstring, best_bleu, num_changed_vars, num_changed_pos, time.time() - time_start)

    @classmethod
    def from_parser_failure(cls, example: Example, time_start: float) -> "Result":
        return cls(example.idx, None, None, None, None, example.code, example.docstring, None, None, None, None, None, time.time() - time_start)


def _tqdm_batch(iterable, n=1):
    len_ = len(iterable)
    for ndx in tqdm(range(0, len_, n)):
        yield iterable[ndx : min(ndx + n, len_)]


def _batch(iterable, n=1):
    len_ = len(iterable)
    for ndx in range(0, len_, n):
        yield iterable[ndx : min(ndx + n, len_)]


def _find_identifiers(code: str) -> list[str]:
    processor = Processor()
    return [i.name.decode() for i in processor.find_identifiers(code)]


def get_unique_id_names(split: str) -> list[str]:
    with multiprocessing.Pool() as pool:
        id_names = {id for id_list in tqdm(pool.map(_find_identifiers, [example.code for example in load_examples(split) + load_examples(f"{split}_ours")])) for id in id_list}
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
    orig_bleu: float
    docstring: str
    replacement: str
    code: str


def compute_scores(totals: dict[str, float], counts: dict[str, int], repl_ids: list[str]) -> list[float]:
    return [totals[repl_id] / counts[repl_id] if repl_id in totals else 0.0 for repl_id in repl_ids]


def _tokenize_batch(batch: list[TransformedCode | Example], tokenizer: AutoTokenizer, args: argparse.Namespace) -> torch.Tensor:
    return tokenizer([x.code for x in batch], padding="max_length", truncation=True, return_tensors="pt", max_length=args.max_source_length).to(args.device)


def _decode_batch(preds: torch.Tensor, tokenizer: AutoTokenizer) -> list[str]:
    preds = torch.where(preds != -100, preds, tokenizer.pad_token_id)
    return tokenizer.batch_decode(preds, skip_special_tokens=True)


def _decode(preds: torch.Tensor, tokenizer: AutoTokenizer) -> str:
    return _decode_batch(preds, tokenizer)[0]


def _tokenize(code: str, tokenizer: AutoTokenizer, args: argparse.Namespace) -> torch.Tensor:
    return tokenizer(code, padding="max_length", truncation=True, return_tensors="pt", max_length=args.max_source_length).to(args.device)


def _bleu_pipeline_batch(model: AutoModelForSeq2SeqLM | Seq2SeqCodeBERT, tokenizer: AutoTokenizer, batch: list[Example | TransformedCode], args: argparse.Namespace) -> tuple[list[float], list[str]]:
    tokenized = _tokenize_batch(batch, tokenizer, args)
    preds = model.generate(**tokenized, max_length=args.max_target_length, num_beams=args.num_beams)
    generations = _decode_batch(preds, tokenizer)
    return compute_bleus(generations, [x.docstring for x in batch]), generations


def _bleu_pipeline(model: AutoModelForSeq2SeqLM | Seq2SeqCodeBERT, tokenizer: AutoTokenizer, code: str, docstring: str, args: argparse.Namespace) -> tuple[float, str]:
    tokenized = _tokenize(code, tokenizer, args)
    preds = model.generate(**tokenized, max_length=args.max_target_length, num_beams=args.num_beams)
    generation = _decode(preds, tokenizer)
    return compute_bleu(generation, docstring), generation


def train(repl_ids: list[str], surrogate: AutoModelForSeq2SeqLM | Seq2SeqCodeBERT, tokenizer: AutoTokenizer, args: argparse.Namespace) -> list[float]:
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
    num_replacements_per_epoch = 20
    best_valid_rate = 0.0
    best_attempt_rate = float("inf")

    print("Getting original BLEU...")
    orig_bleus_map = []
    orig_docstring_map = []
    for batch in tqdm(_batch(train_filtered, n=args.batch_size), total=len(train_filtered) // args.batch_size):
        bleus, docstrings = _bleu_pipeline_batch(surrogate, tokenizer, batch, args)
        orig_bleus_map.extend(bleus)
        orig_docstring_map.extend(docstrings)
        training_attempts += len(batch)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")

        print("Generating transformed code...")
        transformed = []
        for i, example in enumerate(pbar := tqdm(train_filtered)):
            identifiers = find_identifiers_in_trunc_code(example.code, tokenizer, args.max_source_length)
            if len(identifiers) == 0:
                continue
            for identifier in identifiers:
                for replacement in random.sample(filter_repl_ids(repl_ids, identifiers), num_replacements_per_epoch):
                    new_code = apply_transforms(example.code, identifier.to_transforms(replacement.encode()))
                    transformed.append(TransformedCode(orig_bleus_map[i], orig_docstring_map[i], replacement, new_code))

        print("Collecting model feedback...")
        for i, batch in enumerate(pbar := tqdm(_batch(transformed, args.batch_size), total=len(transformed) // args.batch_size)):
            bleus, _ = _bleu_pipeline_batch(surrogate, tokenizer, batch, args)
            training_attempts += len(batch)
            for example, bleu in zip(batch, bleus):
                counts[example.replacement] += 1
                totals[example.replacement] += example.orig_bleu - bleu

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


def get_filtered_examples(examples: list[Example], target: AutoModelForSeq2SeqLM | Seq2SeqCodeBERT, tokenizer: AutoTokenizer, args: argparse.Namespace) -> list[Example]:
    filtered = []
    for batch in _tqdm_batch(examples, BATCH_SIZES[args.model_name]):
        bleus, _ = _bleu_pipeline_batch(target, tokenizer, batch, args)
        filtered.extend([ex for ex, bleu in zip(batch, bleus) if not np.isclose(bleu, 0.0, atol=BLEU_ZERO_ATOL)])
    return filtered


def test_model(args: argparse.Namespace):
    model, tokenizer = initialize_target_model(args)
    examples = load_examples("test")
    all_bleus = []
    all_docstrings = []
    for batch in _batch(examples, args.eval_batch_size):
        bleus, _ = _bleu_pipeline_batch(model, tokenizer, batch, args)
        docstrings = [example.docstring for example in batch]
        all_bleus.extend(bleus)
        all_docstrings.extend(docstrings)
    avg_bleu = sum(all_bleus) / len(all_bleus)
    print(f"Average BLEU: {avg_bleu:.4f}")
    output_path = os.path.join(args.target_weights_dir, "test_results.json")
    print(f"Saving results to {output_path}")
    with open(output_path, "w") as f:
        json.dump({"avg_bleu": avg_bleu}, f)


def truncate_code_to_max_source_length(code: str, tokenizer: AutoTokenizer, max_source_length: int) -> str:
    truncated_tokens = tokenizer.tokenize(code, truncation=True, max_length=max_source_length)[:max_source_length]
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(truncated_tokens))


def find_identifiers_in_trunc_code(code: str, tokenizer: AutoTokenizer, max_source_length: int) -> list[Identifier]:
    processor = Processor()
    trunc_code_len = len(bytes(truncate_code_to_max_source_length(code, tokenizer, max_source_length), "utf-8"))
    ids = processor.find_identifiers(code)
    return [i for i in ids if any(loc.end_byte < trunc_code_len for loc in i.locations)]


def filter_repl_ids(repl_ids: list[str], identifiers: list[Identifier]) -> list[str]:
    id_names = {i.name.decode() for i in identifiers}
    return [repl_id for repl_id in repl_ids if repl_id not in id_names]


def attack_single(target: AutoModelForSeq2SeqLM | Seq2SeqCodeBERT, tokenizer: AutoTokenizer, example: Example, repl_ids: list[str], args: argparse.Namespace, pbar: tqdm, train_attack: bool, quiet: bool = True) -> Result:
    time_start = time.time()
    identifiers = find_identifiers_in_trunc_code(example.code, tokenizer, args.max_source_length)
    if len(identifiers) == 0:
        if not quiet:
            pbar.write("Skipping example with no identifiers")
        return Result.from_parser_failure(example, time_start)
    num_replacement_candidates = 100 if len(identifiers) > 1 else 500

    is_success = get_criterion(args)
    attempts = 0
    orig_bleu, orig_docstring = _bleu_pipeline(target, tokenizer, example.code, example.docstring, args)
    best_bleu = orig_bleu
    best_bleu_by_id = {i: orig_bleu for i in identifiers}
    best_repl_by_id = defaultdict(list)
    best_attempt = 0
    found_new_best = False
    tried_code = dict()

    repl_ids = filter_repl_ids(repl_ids, identifiers)
    replacement_candidates = repl_ids[:num_replacement_candidates] if train_attack else random.sample(repl_ids, k=num_replacement_candidates)
    for round, replacement in enumerate(replacement_candidates, 1):
        for identifier in identifiers:
            new_code = apply_transforms(example.code, identifier.to_transforms(replacement.encode()))
            bleu, new_docstring = _bleu_pipeline(target, tokenizer, new_code, example.docstring, args)
            attempts += 1
            if bleu == (best_bleu := min(best_bleu, bleu)):
                best_attempt = attempts
            if is_success(bleu, orig_bleu):
                if not quiet:
                    pbar.write(f"Successfully attacked example after {attempts} attempts, replaced 1 identifier: {identifier.name.decode()} -> {replacement}: {orig_bleu:.4f} -> {bleu:.4f}")
                return Result.from_success(example, attempts, orig_bleu, orig_docstring, new_code, new_docstring, best_bleu, 1, len(identifier.locations), time_start)
            if bleu < best_bleu_by_id[identifier]:
                found_new_best = True
                best_bleu_by_id[identifier] = bleu
                best_repl_by_id[identifier].append((replacement, bleu))
                if not quiet:
                    pbar.write(f"Best prob for {identifier.name.decode()}: {replacement}: {orig_bleu:.4f} -> {best_bleu_by_id[identifier]:.4f}")

        if round % 10 == 0 and found_new_best and sum(bleu != orig_bleu for bleu in best_bleu_by_id.values()) > 1 and len(identifiers) > 1:
            if not quiet:
                pbar.write(f"Round {round}: Attempted {attempts} replacements, best bleu: {best_bleu:.4f} (after {best_attempt}/{attempts} attempts)")
            best_bleu = orig_bleu
            best_bleu_id_repl = sorted([(i, r, p) for i in identifiers for (r, p) in best_repl_by_id[i]], key=lambda x: x[2])
            used_transforms = []
            used_identifiers = set()
            used_replacements = set()
            for identifier, replacement, _ in best_bleu_id_repl:
                if identifier in used_identifiers or replacement in used_replacements:
                    continue
                new_transforms = used_transforms + identifier.to_transforms(replacement.encode())
                new_code = apply_transforms(example.code, new_transforms)
                if new_code not in tried_code:
                    tried_code[new_code] = _bleu_pipeline(target, tokenizer, new_code, example.docstring, args)
                    attempts += 1
                bleu, new_docstring = tried_code[new_code]
                if bleu < best_bleu:
                    used_identifiers.add(identifier)
                    used_replacements.add(replacement)
                    used_transforms = new_transforms
                    if not quiet:
                        pbar.write(f"Combining {identifier.name.decode()} -> {replacement} with existing transforms improved prob: {best_bleu:.4f} -> {bleu:.4f}")
                else:
                    if not quiet:
                        pbar.write(f"Combining {identifier.name.decode()} -> {replacement} with existing transforms did not improve prob: {best_bleu:.4f} -> {bleu:.4f}")
                if bleu == (best_bleu := min(best_bleu, bleu)):
                    best_attempt = attempts
                if is_success(bleu, orig_bleu):
                    if not quiet:
                        pbar.write(f"Successfully attacked example after {attempts} attempts, replaced {len(used_identifiers)} identifiers: {', '.join(i.name.decode() for i in used_identifiers)}")
                    return Result.from_success(example, attempts, orig_bleu, orig_docstring, new_code, new_docstring, best_bleu, len(used_identifiers), sum(len(i.locations) for i in used_identifiers), time_start)
            found_new_best = False

    if not quiet:
        pbar.write(
            f"Failed to attack example: orig prob: {orig_bleu:.4f}, best prob: {best_bleu:.4f} (after {best_attempt}/{attempts} attempts), num_identifiers: {len(identifiers)}, num_attempts: {attempts}"
        )
    return Result.from_failure(example, attempts, orig_bleu, orig_docstring, best_bleu, time_start)


class AttackStats(BaseAttackStats):
    def __init__(self):
        super().__init__()
        self.bleu_drop_percent = 0
        self.bleu_drop = 0

    def update(self, result: Result):
        super().update(result)
        if result.success:
            self.bleu_drop_percent += 100 * (result.orig_bleu - result.best_bleu) / result.orig_bleu
            self.bleu_drop += result.orig_bleu - result.best_bleu
        self.bleu_drop_percent_rate = self.bleu_drop_percent / self.successes if self.successes else 0
        self.bleu_drop_rate = 100 * self.bleu_drop / self.successes if self.successes else 0

    def __str__(self):
        return super().__str__() + ", " \
               f"BP: {self.bleu_drop_percent_rate:.2f}, " \
               f"BD: {self.bleu_drop_rate:.2f}, " \


BATCH_SIZES = {
    "microsoft/codebert-base": 256,
    "microsoft/graphcodebert-base": 256,
    "Salesforce/codet5p-220m": 50,
}


def parse_args():
    parser = common.ArgumentParser()
    parser.add_argument("--from_scratch", action="store_true", help="Rewrite cached data")
    parser.add_argument("--allow_multi_gpu", action="store_true", help="Allow multi-gpu training when available")
    args = parser.parse_args()
    args.batch_size = BATCH_SIZES[args.model_name]
    os.makedirs(args.results_dir, exist_ok=True)
    return args


def attack(args: argparse.Namespace, train_attack: bool):
    target, tokenizer = initialize_target_model(args)
    print("Preparing attack...")
    repl_ids = preprocess(args)
    random.seed(args.seed)
    repl_ids = random.sample(repl_ids, k=min(20000, len(repl_ids)))
    if train_attack:
        print("Training attacker...")
        scores = train(repl_ids, target, tokenizer, args)
    print("Starting attack...")
    examples = load_examples("test")
    print("Filtering examples...")
    filtered_examples = get_filtered_examples(examples, target, tokenizer, args)
    if train_attack:
        # Sort repl_ids by scores
        repl_ids = [repl_ids[i] for i in np.argsort(scores)[::-1]]
    else:
        random.seed(args.seed)

    stats = AttackStats()
    with open(args.results_path, "w") as f:
        for i, example in enumerate(pbar := tqdm(filtered_examples), 1):
            pbar.write(f"Attacking example {i}:")
            result = attack_single(target, tokenizer, example, repl_ids, args, pbar, train_attack, False)
            stats.update(result)
            pbar.set_description(str(stats))
            f.write(json.dumps(result._asdict()) + "\n")
    print_results(args.results_path)


if __name__ == "__main__":
    with torch.no_grad():
        attack(parse_args(), train_attack=False)
