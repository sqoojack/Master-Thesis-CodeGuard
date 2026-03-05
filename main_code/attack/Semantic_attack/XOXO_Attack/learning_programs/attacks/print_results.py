import argparse
import json
import os
from typing import NamedTuple, Optional

from learning_programs.attacks.common import AttackConfig, SUPPORTED_MODELS, LANGUAGES
from learning_programs.metrics.codebleu import codebleu

class Result(NamedTuple):
    idx: int
    success: Optional[bool]
    queries: Optional[int]
    code: str
    adv_code: Optional[str]
    target: int
    num_changed_vars: Optional[int]
    num_changed_pos: Optional[int]
    time: Optional[float]


class SummarizationResult(NamedTuple):
    idx: int
    success: Optional[bool]
    queries: Optional[int]
    orig_bleu: str
    orig_docstring: str
    code: str
    docstring: str
    adv_code: Optional[str]
    adv_docstring: Optional[str]
    best_bleu: Optional[float]
    num_changed_vars: Optional[int]
    num_changed_pos: Optional[int]
    time: Optional[float]


class GenerationResult(NamedTuple):
    task_id: int
    success: Optional[bool]
    queries: Optional[int]
    batch_queries: Optional[int]
    share_tests_passed: Optional[float]
    code: str
    adv_code: Optional[str]
    generated_code: Optional[str]
    identifier: str
    replacement: str
    num_changed_vars: Optional[int]
    num_changed_pos: Optional[int]
    time: Optional[float]


def print_results(results_path: str, compute_codebleu: bool = False, dataset: str = None):
    if not os.path.exists(results_path):
        print("No results found")
        return
    with open(results_path) as f:
        if "summarization" in results_path:
            results = [SummarizationResult(**json.loads(line)) for line in f]
        elif any(x in results_path for x in ["cweval", "mbpp", "humaneval"]):
            results = [GenerationResult(**json.loads(line)) for line in f]
        else:
            results = [Result(**json.loads(line)) for line in f]
    num_successes = sum(r.success for r in results if r.success is not None)
    num_queries = sum(r.queries for r in results if r.success is not None)
    num_no_na = len([r for r in results if r.success is not None])
    num_changed_vars = sum(r.num_changed_vars for r in results if r.success)
    num_changed_pos = sum(r.num_changed_pos for r in results if r.success)
    accuracy = num_successes / len(results)
    accuracy_no_na = num_successes / num_no_na
    total_time_mins = sum(r.time for r in results) / 60

    print(f"Success rate: {accuracy:.2%}")
    print(f"Success rate (no NAs): {accuracy_no_na:.2%}")
    if "summarization" in results_path:
        print(f"Average BLEU drop: {sum(r.orig_bleu - r.best_bleu for r in results if r.success) / num_successes:.2f}")
        print(f"Average BLEU before attack: {sum(r.orig_bleu for r in results if r.success) / num_successes:.2f}")
        print(f"Average BLEU after attack: {sum(r.best_bleu for r in results if r.success) / num_successes:.2f}")
    if compute_codebleu:
        if dataset is None:
            raise ValueError("Dataset must be provided to compute codebleu")
        codes = [r.code for r in results if r.success]
        adv_codes = [r.adv_code for r in results if r.success]
        score = codebleu.compute_codebleu(codes, adv_codes, LANGUAGES[dataset])
        print(f"CodeBLEU: {score*100:.2f}")
    print(f"Average queries: {num_queries / num_no_na:.2f}")
    print(f"Average number of changed variables: {num_changed_vars / num_successes:.2f}")
    print(f"Average number of changed positions: {num_changed_pos / num_successes:.2f}")
    print(f"Total time: {total_time_mins:.2f} minutes")
    print(f"Average time: {total_time_mins / num_successes:.2f} minutes")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=SUPPORTED_MODELS.keys(), required=True)
    parser.add_argument("--attack", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--compute_codebleu", action="store_true")
    args = parser.parse_args()
    if args.model_name not in SUPPORTED_MODELS[args.dataset]:
        raise ValueError(f"Model {args.model_name} not supported for dataset {args.dataset}")
    return args


if __name__ == "__main__":
    args = parse_args()
    print_results(AttackConfig(args.dataset, args.attack).get_results_path(args.model_name, args.seed), args.compute_codebleu, args.dataset)
