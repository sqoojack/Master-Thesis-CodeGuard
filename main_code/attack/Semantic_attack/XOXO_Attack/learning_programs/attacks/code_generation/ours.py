import argparse
import json
import multiprocessing
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from learning_programs.attacks import common
from learning_programs.attacks.common import PREPROCESS_SEED
from learning_programs.attacks.print_results import GenerationResult, print_results
from learning_programs.attacks.code_generation.api import OpenAIConnector, SonnetConnector
from learning_programs.attacks.code_generation.prompts import prompt_template
from learning_programs.attacks.defect_detection.ours import AttackStats, compute_scores, filter_repl_ids
from learning_programs.datasets.code_generation import load_examples, Example
from learning_programs.datasets.summarization import load_examples as load_examples_codesearchnet
from learning_programs.transforms.processor import apply_transforms
from learning_programs.transforms.transform import Identifier, Transform
from learning_programs.transforms.tree_utils import get_nodes
from learning_programs.transforms.python.processor import Processor, PY_PARSER
from learning_programs.runners.docker.runner import TestRunnerManager as TRMDocker
from learning_programs.runners.python.runner import TestRunnerManager as TRMPython
from learning_programs.runners.python.runner import TestRunnerManager, TestResult, SA_NO_ENTRY_POINT, SA_MAGIC_PARSER_ERROR


PASS_1 = SamplingParams(n=1, temperature=0, max_tokens=2048, logprobs=1, stop="```")

NUM_SAMPLED_IDS = {
    "humaneval": 10000,
    "mbpp": 2000,
    "cweval": 100000
}


def get_unique_id_names(dataset: str) -> list[str]:
    processor = Processor()
    id_names = set()
    for example in tqdm(load_examples_codesearchnet("train")):
        id_names |= {i.name.decode() for i in processor.find_identifiers(example.code)}
    random.seed(PREPROCESS_SEED)
    id_names = random.sample(sorted(id_names), NUM_SAMPLED_IDS[dataset])
    return sorted(id_names) # Randomly sample 10000 identifiers, sorted for reproducibility


def preprocess(args: argparse.Namespace) -> list[str]:
    save_path_ids = os.path.join(args.results_dir, "ids.json")

    if not args.from_scratch and os.path.exists(save_path_ids):
        print(f"Found existing data at {args.results_dir}, loading...")
        with open(save_path_ids) as f:
            return json.load(f)

    print("Scraping training data for identifiers...")
    id_names = get_unique_id_names(args.dataset)
    with open(save_path_ids, "w") as f:
        json.dump(id_names, f)
    print(f"Saved unique identifier names to {save_path_ids}")
    return id_names


LIMITED_GPU = False


class Model:
    llm: LLM
    tokenizer: AutoTokenizer
    tmgr: TestRunnerManager
    num_passes: int
    queries: int

    def __init__(self, model: str, tmgr: TestRunnerManager, allow_multi_gpu: bool = False):
        tensor_parallel_size = torch.cuda.device_count() if allow_multi_gpu else 1
        if LIMITED_GPU:
            self.llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=0.95,
                max_model_len=8192,
                max_num_seqs=64
            )
        elif tmgr:
            self.llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                tokenizer_pool_size=tmgr.max_workers
            )
        else:
            self.llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
            )
        self.queries = 0
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, padding_side="left")
        self.tmgr = tmgr
    
    def finalize_prompt(self, prompt: str):
        return prompt_template(prompt.replace('    ', '\t'), self.tokenizer)

    def generate(self, examples: list[Example]) -> tuple[list[str], list[list[str]], list[list[float]]]:
        self.queries += len(examples)
        prompts = [self.finalize_prompt(example.prompt) for example in examples]
        req_outputs = self.llm.generate(prompts, sampling_params = PASS_1, use_tqdm = True)
        generated_code = [[process_llm_resp(o.text) for o in r.outputs] for r in req_outputs]
        mean_logprobs = [[o.cumulative_logprob / len(o.logprobs) for o in r.outputs] for r in req_outputs]
        return prompts, generated_code, mean_logprobs

    def generate_test(self, examples: list[Example]) -> tuple[list[str], list[list[str]], list[list[TestResult]], list[list[float]]]:
        prompts, generations, cum_logprobs = self.generate(examples)
        tasks = [example.task for example in examples]
        processed_generations = [[preprocess_generation(example, code) for code in generation] for example, generation in zip(examples, generations)]
        return prompts, generations, self.tmgr.test(tasks, processed_generations), cum_logprobs
    
    def generate_preprocess(self, examples: list[Example]) -> tuple[list[str], list[list[str]], list[list[str]], list[list[float]]]:
        prompts, generations, cum_logprobs = self.generate(examples)
        processed_generations = [[preprocess_generation(example, code) for code in generation] for example, generation in zip(examples, generations)]
        return prompts, generations, processed_generations, cum_logprobs


class APIModel(Model):
    def __init__(self, model:str, tmgr: TestRunnerManager):
        self.model = model
        self.num_workers = 100
        if self.model == "openai/gpt-4.1":
            self.connector = OpenAIConnector(self.model, "openrouter")
        elif "sonnet" in self.model:
            self.connector = SonnetConnector(self.model)
        else:
            raise ValueError(f"Unsupported API model: {self.model}")
        self.queries = 0
        self.tokenizer = None # API models tokenize internally
        self.tmgr = tmgr

    def generate(self, examples: list[Example]) -> tuple[list[str], list[list[str]], list[list[float]] | None]:
        self.queries += len(examples)
        prompts = [self.finalize_prompt(example.prompt) for example in examples]

        batched_prompts = [prompts[i::self.num_workers] for i in range(self.num_workers)]
        places = [list(range(i, len(prompts), self.num_workers)) for i in range(self.num_workers)]
        ret = [None] * len(prompts)

        with tqdm(desc="Processed prompts", total=len(prompts)) as pbar, ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            def wait_for_results(i: int) -> None:
                for prompt, place in zip(batched_prompts[i], places[i]):
                    ret[place] = self.connector.get_response(prompt)
                    pbar.update()

            futures = [executor.submit(wait_for_results, i) for i in range(self.num_workers)]
            for future in as_completed(futures):
                future.result()

        generated_code = [[process_llm_resp(o['message']['content']) for o in r['choices']] for r in ret]
        cum_logprobs = [[sum(logprob['logprob'] for logprob in o['logprobs']) / len(o['logprobs']) for o in r['choices']] for r in ret] if type(self.connector) is OpenAIConnector else None
        return prompts, generated_code, cum_logprobs


def process_llm_resp(code: str) -> str:
    if '```python' in code:
        code = code.split('```python')[-1]
        code = code.split('```')[0]
    elif '```Python' in code:
        code = code.split('```Python')[-1]
        code = code.split('```')[0]
    elif '```' in code:
        code = code.split('```')[0]
    code = code.strip()
    return code


def preprocess_generation(example: Example, code: str) -> str:
    # Check for syntax errors
    tree = PY_PARSER.parse(code.encode())
    if any(node.is_error or node.is_missing for node in get_nodes(tree)):
        return SA_MAGIC_PARSER_ERROR + code
    identifiers = Processor().find_identifiers(code)
    if len(identifiers) == 0:
        return SA_MAGIC_PARSER_ERROR + code
    # Entry point is missing
    if not any(i.name.decode() == example.entry_point for i in identifiers):
        return SA_NO_ENTRY_POINT + code
    # Revert the entry point if it was replaced
    if example.entry_point != example.task["entry_point"]:
        for identifier in identifiers:
            if identifier.name.decode() == example.entry_point:
                return apply_transforms(code, identifier.to_transforms(example.task["entry_point"].encode()))
    return code


def get_filtered_examples(examples: list[Example], target: Model) -> list[Example]:
    is_cweval = examples[0].task["task_id"].startswith("CWEval")
    check_result = lambda result: result[0].functional and result[0].secure if is_cweval else result[0].passed
    _, _, results, _ = target.generate_test(examples)
    filtered = [example for example, result in zip(examples, results) if check_result(result)]
    return filtered


def get_share_passed(results: list[TestResult]) -> float:
    return sum(res.passed for res in results) / len(results)


class SingleTransformedExample(Example):
    identifier: Identifier
    replacement: str

    def __init__(self, example: Example, new_prompt: str, new_entry_point: str, identifier: Identifier, replacement: str):
        super().__init__(new_prompt, new_entry_point, example.task, example.problems_entry_points)
        self.identifier = identifier
        self.replacement = replacement

    @classmethod
    def apply_transforms(cls, example: Example, identifier: Identifier, replacement: str) -> "SingleTransformedExample":
        new_prompt = apply_transforms(example.prompt, identifier.to_transforms(replacement.encode()))
        id_name = identifier.name.decode()
        # Replace the identifier in docstrings
        if id_name in example.problems_entry_points:
            new_prompt = new_prompt.replace(f"{id_name}(", f"{replacement}(")
        # Replace the entry point if the corresponding identifier was replaced
        new_entry_point = example.entry_point
        if id_name == example.entry_point:
            new_entry_point = replacement
        return cls(example, new_prompt, new_entry_point, identifier, replacement)
    
    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "entry_point": self.entry_point,
            "problems_entry_points": list(self.problems_entry_points),
            "identifier": self.identifier.name.decode(),
            "num_identifiers": 1,
            "num_locations": len(self.identifier.locations),
            "replacement": self.replacement
        }


class MultiTransformedExample(Example):
    identifiers: Identifier
    replacements: str

    def __init__(self, example: Example, new_prompt: str, new_entry_point: str, identifiers: list[Identifier], replacements: list[str]):
        super().__init__(new_prompt, new_entry_point, example.task, example.problems_entry_points)
        self.identifiers = identifiers
        self.replacements = replacements

    @classmethod
    def apply_transforms(cls, example: Example, identifiers: list[Identifier], replacements: list[str]) -> "MultiTransformedExample":
        transforms = []
        for identifier, replacement in zip(identifiers, replacements):
            transforms.extend(identifier.to_transforms(replacement.encode()))
        new_prompt = apply_transforms(example.prompt, transforms)
        new_entry_point = example.entry_point
        for identifier, replacement in zip(identifiers, replacements):
            id_name = identifier.name.decode()
            # Replace the identifier in docstrings
            if id_name in example.problems_entry_points:
                new_prompt = new_prompt.replace(f"{id_name}(", f"{replacement}(")
            # Replace the entry point if the corresponding identifier was replaced
            if id_name == example.entry_point:
                new_entry_point = replacement
        return cls(example, new_prompt, new_entry_point, identifiers, replacements)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "entry_point": self.entry_point,
            "problems_entry_points": list(self.problems_entry_points),
            "identifiers": [i.name.decode() for i in self.identifiers],
            "num_identifiers": len(self.identifiers),
            "num_locations": sum(len(i.locations) for i in self.identifiers),
            "replacements": self.replacements
        }


class Result(GenerationResult):
    @classmethod
    def from_failure(cls, example: Example, queries: int, batch_queries: int, time_start: float) -> "Result":
        return cls(example.task["task_id"], False, queries, batch_queries, None, example.prompt, None, None, None, None, None, None, time.time() - time_start)

    @classmethod
    def from_success(cls, example: Example, queries: int, batch_queries: int, generated_code: str, share_tests_passed: float, tr_example: SingleTransformedExample | MultiTransformedExample, time_start: float) -> "Result":
        if type(tr_example) is SingleTransformedExample:
            return cls(example.task["task_id"], True, queries, batch_queries, share_tests_passed, example.prompt, tr_example.prompt, generated_code, tr_example.identifier.name.decode(), tr_example.replacement, 1, len(tr_example.identifier.locations), time.time() - time_start)
        else:
            id_names = [i.name.decode() for i in tr_example.identifiers]
            num_locations = sum(len(i.locations) for i in tr_example.identifiers)
            return cls(example.task["task_id"], True, queries, batch_queries, share_tests_passed, example.prompt, tr_example.prompt, generated_code, id_names, tr_example.replacements, len(tr_example.identifiers), num_locations, time.time() - time_start)


def train(repl_ids: list[str], surrogate: Model, args: argparse.Namespace) -> list[float]:
    time_start = time.time()
    save_path = os.path.join(args.results_dir, "scores.json")

    if not args.from_scratch and os.path.exists(save_path):
        print(f"Found existing scores at {save_path}, loading...")
        with open(save_path) as f:
            return json.load(f)

    train_dataset = "mbpp" if args.dataset == "humaneval" else "humaneval"
    examples = load_examples(train_dataset)

    print(f"Setting seed to {args.seed}")
    random.seed(args.seed)

    print("Filtering train and validation examples...")
    examples_filtered = get_filtered_examples(examples, surrogate)
    print(f"Filtered {len(examples_filtered)}/{len(examples)} examples that the target model got right")
    valid_filtered = random.sample(examples, 20)
    valid_example_ids = {ex.task["task_id"] for ex in valid_filtered}
    train_filtered = [ex for ex in examples if ex.task["task_id"] not in valid_example_ids]
    counts = defaultdict(int)
    totals = defaultdict(float)
    training_attempts = 0
    num_epochs = 10
    num_replacements_per_epoch = 20 if train_dataset == "mbpp" else 40
    best_valid_rate = 0.0
    best_attempt_rate = float("inf")

    print("Getting original probabilities...")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}")

        print("Generating transformed code...")
        transformed = []
        for example in tqdm(train_filtered):
            identifiers = find_identifiers(example)
            if len(identifiers) == 0:
                continue
            for identifier in identifiers:
                for replacement in random.sample(filter_repl_ids(repl_ids, identifiers), num_replacements_per_epoch):
                    transformed.append(SingleTransformedExample.apply_transforms(example, identifier, replacement))

        print("Collecting model feedback...")
        _, _, results, _ = surrogate.generate_test(transformed)
        training_attempts += len(transformed)
        for example, result in zip(transformed, results):
            result = result[0] # pass 1
            counts[example.replacement] += 1
            totals[example.replacement] += not result.timed_out and not result.passed

        print(f"Top 5 best performing identifiers after {training_attempts} transforms:")
        for rank, k in enumerate(sorted(totals, key=lambda k: totals[k] / counts[k], reverse=True)[:5], 1):
            print(f"{rank}: {k}: {totals[k] / counts[k]:.4f} ({counts[k]})")

        print("Validating...")
        scores = compute_scores(totals, counts, repl_ids)
        ranked_repl_ids = [repl_ids[i] for i in np.argsort(scores)[::-1]]
        stats = AttackStats()
        for example in (pbar := tqdm(valid_filtered)):
            result = attack_single(surrogate, example, ranked_repl_ids, pbar, True, True)
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


class StringTransform(Transform):
    def __init__(self, start: int, end: int, replace: bytes):
        self.start = start
        self.end = end
        self.result = replace
        self.name = type(self).__name__


class StringIdentifier(Identifier):
    def __init__(self, name: str, text: str):
        self.name = name.encode()
        self.locations = [m.span() for m in re.finditer(self.name, text.encode())]
        self.origin = "string_entry_point"

    @classmethod
    def from_example(cls, example: Example):
        return cls(example.entry_point, example.prompt)

    def to_transforms(self, replacement: bytes) -> list[Transform]:
        return [StringTransform(start, end, replacement) for start, end in self.locations]


def find_identifiers(example: Example) -> list[Identifier]:
    ids = Processor().find_identifiers(example.prompt)
    if example.task["task_id"].startswith("Mbpp"):
        ids += [StringIdentifier(example.entry_point, example.prompt)]
    return ids


def attack_single(target: Model, example: Example, repl_ids: list[str], pbar: tqdm, train_attack: bool, use_logprobs: bool, quiet: bool = True) -> Result:
    if use_logprobs:
        return attack_single_combs(target, example, repl_ids, pbar, train_attack, quiet)
    return attack_single_no_combs(target, example, repl_ids, pbar, train_attack, quiet)


def perplexity(mean_logprob: float) -> float:
    return np.exp(-mean_logprob)


def get_num_replacement_candidates_per_identifier(target: Model, example: Example) -> int:
    if example.task["task_id"].startswith("CWEval"):
        return 500
    if type(target) is APIModel and target.model == "openai/gpt-4.1":
        if example.task["task_id"].startswith("HumanEval"):
            return 30
        return 20
    return 50


def get_test_every_per_identifier(target: Model, example: Example) -> int:
    if type(target) is APIModel and not example.task["task_id"].startswith("CWEval"):
        if target.model == "openai/gpt-4.1" and example.task["task_id"].startswith("HumanEval"):
            return 3
        return 1
    return 5


def attack_single_combs(target: Model, example: Example, repl_ids: list[str], pbar: tqdm, train_attack: bool, quiet: bool = True) -> Result:
    time_start = time.time()
    identifiers = find_identifiers(example)
    target.queries = 0
    num_replacement_candidates = get_num_replacement_candidates_per_identifier(target, example) * len(identifiers)
    transformed_examples = []
    test_every = get_test_every_per_identifier(target, example) * len(identifiers) # No need to batch for API models
    attempts = 0
    mean_logprobs_map = dict()
    already_tried = set()

    repl_ids = filter_repl_ids(repl_ids, identifiers)
    replacement_candidates = repl_ids[:num_replacement_candidates] if train_attack else random.sample(repl_ids, k=num_replacement_candidates)
    for round in range(1, num_replacement_candidates + 1):
        for identifier, replacement in zip(identifiers, replacement_candidates[(round - 1) * len(identifiers):round * len(identifiers)]):
            transformed_examples.append(SingleTransformedExample.apply_transforms(example, identifier, replacement))

        if round < num_replacement_candidates and len(transformed_examples) < test_every:
            continue

        prompts, generations, results, mean_logprobs = target.generate_test(transformed_examples)

        for tr_example, prompt, generation, result, mean_logprob in zip(transformed_examples, prompts, generations, results, mean_logprobs):
            generation = generation[0] # pass 1
            result = result[0] # pass 1
            mean_logprob = mean_logprob[0] # pass 1
            mean_logprobs_map[(tr_example.identifier, tr_example.replacement)] = mean_logprob
            if not quiet:
                pbar.write(f"{tr_example.identifier.name.decode()} -> {tr_example.replacement}: {result}: perplexity={perplexity(mean_logprob):.4f}")
                # pbar.write(str(prompt))
                # pbar.write("-"*80)
                # pbar.write(generation)
            attempts += 1
            if not result.passed and not result.timed_out:
                if not quiet:
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
            if not quiet:
                pbar.write(f"{identifier.name.decode()} -> {replacement}: perplexity={perplexity(mean_logprobs_map[(identifier, replacement)]):.4f}")
            if (tuple(used_identifiers), tuple(used_replacements)) not in already_tried:
                transformed_examples.append(MultiTransformedExample.apply_transforms(example, used_identifiers.copy(), used_replacements.copy()))
                already_tried.add((tuple(used_identifiers), tuple(used_replacements)))

        prompts, generations, results, mean_logprobs = target.generate_test(transformed_examples)
        for tr_example, prompt, generation, result, mean_logprobs in zip(transformed_examples, prompts, generations, results, mean_logprobs):
            generation = generation[0] # pass 1
            result = result[0] # pass 1
            mean_logprob = mean_logprobs[0] # pass 1
            if not quiet:
                pbar.write(f"{str([i.name.decode() for i in tr_example.identifiers])} -> {str([r for r in tr_example.replacements])}: {result}: perplexity={perplexity(mean_logprob):.4f}")
            attempts += 1
            if not result.passed and not result.timed_out:
                if not quiet:
                    pbar.write(f"Successfully attacked example after {attempts} attempts, replaced {len(tr_example.identifiers)} identifiers.")
                    pbar.write(str(prompt))
                    pbar.write("-"*80)
                    pbar.write(generation)
                return Result.from_success(example, attempts, target.queries, generation, result.share_tests_passed, tr_example, time_start)

        # if type(target) is not APIModel: # No need to batch for API models
        if type(target) is not APIModel or (target.model == "openai/gpt-4.1" and example.task["task_id"].startswith("HumanEval")): # No need to batch for API models
            test_every *= 2
        transformed_examples = []

    if not quiet:
        pbar.write(f"Failed to attack example after {target.queries} attempts.")
    return Result.from_failure(example, attempts, target.queries, time_start)


def attack_single_no_combs(target: Model, example: Example, repl_ids: list[str], pbar: tqdm, train_attack: bool, quiet: bool = True) -> Result:
    time_start = time.time()
    identifiers = find_identifiers(example)
    target.queries = 0
    num_replacement_candidates = get_num_replacement_candidates_per_identifier(target, example) * len(identifiers)
    transformed_examples = []
    test_every = get_test_every_per_identifier(target, example) * len(identifiers) # No need to batch for API models
    attempts = 0

    repl_ids = filter_repl_ids(repl_ids, identifiers)
    replacement_candidates = repl_ids[:num_replacement_candidates] if train_attack else random.sample(repl_ids, k=num_replacement_candidates)
    for round in range(1, num_replacement_candidates + 1):
        for identifier, replacement in zip(identifiers, replacement_candidates[(round - 1) * len(identifiers):round * len(identifiers)]):
            transformed_examples.append(SingleTransformedExample.apply_transforms(example, identifier, replacement))

        if round < num_replacement_candidates and len(transformed_examples) < test_every:
            continue

        prompts, generations, results, _ = target.generate_test(transformed_examples)

        for tr_example, prompt, generation, result in zip(transformed_examples, prompts, generations, results):
            generation = generation[0] # pass 1
            result = result[0] # pass 1
            if not quiet:
                pbar.write(f"{tr_example.identifier.name.decode()} -> {tr_example.replacement}: {result}")
            attempts += 1
            if not result.passed and not result.timed_out:
                if not quiet:
                    pbar.write(f"Successfully attacked example after {attempts} attempts, replaced 1 identifier: {tr_example.identifier.name.decode()} -> {tr_example.replacement}")
                    pbar.write(str(prompt))
                    pbar.write("-"*80)
                    pbar.write(generation)
                return Result.from_success(example, attempts, target.queries, generation, result.share_tests_passed, tr_example, time_start)

        if type(target) is not APIModel: # No need to batch for API models
            test_every *= 2
        transformed_examples = []

    if not quiet:
        pbar.write(f"Failed to attack example after {target.queries} attempts.")
    return Result.from_failure(example, attempts, target.queries, time_start)


def parse_args():
    parser = common.ArgumentParser()
    parser.add_argument("--from_scratch", action="store_true", help="Rewrite the cache")
    parser.add_argument("--allow_multi_gpu", action="store_true", help="Allow multi-gpu training when available")
    parser.add_argument("--max_workers", type=int, default=multiprocessing.cpu_count(), help="Maximum number of workers to use")
    return parser.parse_args()


def attack(args: argparse.Namespace, train_attack: bool, use_logprobs: bool):
    if use_logprobs and "sonnet" in args.model_name:
        raise ValueError("Logprobs are not supported for Sonnet models")
    print("Loading target model...")

    with (TRMDocker if args.dataset == "cweval" else TRMPython)(args.max_workers) as tmgr:
        if args.model_name in common.CODE_GENERATION_API_MODELS:
            target = APIModel(args.model_name, tmgr)
        else:
            target = Model(args.model_name, tmgr, args.allow_multi_gpu)
        print("Preparing attack...")
        repl_ids = preprocess(args)
        test_examples = load_examples(args.dataset)
        print("Filtering examples...")
        filtered = get_filtered_examples(test_examples, target)
        print(f"Filtered {len(filtered)}/{len(test_examples)} examples that the target model got right")

        if train_attack:
            print("Training attack...")
            scores = train(repl_ids, target, args)

        print(f"Setting seed to {args.seed}")
        random.seed(args.seed)

        print("Starting attack...")
        if train_attack:
            repl_ids = [repl_ids[i] for i in np.argsort(scores)[::-1]]

        stats = AttackStats()
        with open(args.results_path, "w") as f:
            for i, example in enumerate(pbar := tqdm(filtered), 1):
                pbar.write(f"Attacking example {i}: {example.task['task_id']} / {example.task['entry_point']}")
                result = attack_single(target, example, repl_ids, pbar, train_attack, use_logprobs, False)
                stats.update(result)
                pbar.set_description(str(stats))
                f.write(json.dumps(result._asdict()) + "\n")
        print_results(args.results_path)
