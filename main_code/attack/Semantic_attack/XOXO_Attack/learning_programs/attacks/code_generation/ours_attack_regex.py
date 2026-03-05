import argparse
import os
import random

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from learning_programs.attacks.defect_detection.ours import filter_repl_ids
from learning_programs.transforms.processor import apply_transforms
from learning_programs.transforms.python.processor import Processor
from learning_programs.runners.python.runner import TestRunnerManager

from learning_programs.attacks.code_generation.ours import get_unique_id_names, PASS_1


def preprocess() -> list[str]:
    print("Scraping training data for identifiers...")
    return get_unique_id_names()


def zero_shot_with_context_template(code_snippet: str, tokenizer: AutoTokenizer):
    _MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"
    prompt = f'''Complete the following Python code snippet:
{code_snippet.strip()}
'''
    response = f"""Below is a continuation of the Python code that solves the problem.```python{_MAGIC_SPLITTER_}```"""
    input = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]
            
    return input


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


class Model:
    engine: LLM
    tokenizer: AutoTokenizer
    tmgr: TestRunnerManager

    def __init__(self, model: str, tmgr: TestRunnerManager, allow_multi_gpu: bool = False):
        tensor_parallel_size = torch.cuda.device_count() if allow_multi_gpu else 1
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.95,
            max_model_len=16384,
            max_num_seqs=64
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast = True, padding_side = "left")
        self.tmgr = tmgr
    
    def finalize_prompt(self, prompt: str):
        return zero_shot_with_context_template(prompt.replace('    ', '\t'), self.tokenizer)

    def generate(self, codes: list[str], params: SamplingParams, use_tqdm: bool = True) -> list[list[str]]:
        prompts = [self.finalize_prompt(c) for c in codes]
        req_outputs = self.llm.generate(prompts, sampling_params=params, use_tqdm = use_tqdm)
        generated_code = [[process_llm_resp(o.text) for o in r.outputs] for r in req_outputs]
        return generated_code


def attack_single_regex(target: Model, target_code: str, target_substring: str, repl_ids: list[str], results_dir: str):
    identifiers = Processor().find_identifiers(target_code)
    if not identifiers:
        print("No identifiers found in target code.")
        return
    attempts = 0
    num_replacement_candidates_per_identifier = 10
    round = 1

    repl_ids = filter_repl_ids(repl_ids, identifiers)
    try:
        while True:
            print(f"Round {round}")
            replacement_candidates = random.sample(repl_ids, num_replacement_candidates_per_identifier * len(identifiers))
            codes = []
            replacements = []
            for identifier in identifiers:
                for replacement in replacement_candidates:
                    codes.append(apply_transforms(target_code, identifier.to_transforms(replacement.encode())))
                    replacements.append((identifier, replacement))

            generations = target.generate(codes, PASS_1)

            for gens, (identifier, replacement) in zip(generations, replacements):
                print(f"{identifier.name.decode()} -> {replacement}")
                for gen in gens:
                    print(gen)
                    attempts += 1
                    if target_substring in gen:
                        print(f"Successfully attacked example after {attempts} attempts, replaced 1 identifier: {identifier.name.decode()} -> {replacement}")
                        save_path = os.path.join(results_dir, f"{round}_{attempts}_from={identifier.name.decode()}_to={replacement}.txt")
                        print(f"Saving to {save_path}")
                        with open(save_path, "w") as f:
                            f.write(gen)
            round += 1

    except KeyboardInterrupt:
        print(f"Aborted after {target.queries} queries.")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Run an attack on a model")
    parser.add_argument("--model_name", type=str, help="Name of the model to attack", default="Qwen/CodeQwen1.5-7B-Chat")
    parser.add_argument("--seed", type=int, help="Seed for the random number generator", default=2024)
    parser.add_argument("--allow_multi_gpu", action="store_true", help="Allow multi-gpu training when available")
    parser.add_argument("--target_snippet_file", type=str, help="Path to the target snippet file", required=True)
    parser.add_argument("--target_substring", type=str, help="Substring to replace in the target snippet", required=True)
    parser.add_argument("--results_dir", type=str, help="Path to the results directory", default="results/ours_attack_regex")
    args = parser.parse_args()
    args.results_dir = os.path.join(args.results_dir, args.model_name, str(args.seed))
    return parser.parse_args()


def attack(args: argparse.Namespace):
    print("Loading target model...")
    with TestRunnerManager() as tmgr:
        target = Model(args.model_name, tmgr, args.allow_multi_gpu)
        print("Preparing attack...")
        repl_ids = preprocess()

        print(f"Setting seed to {args.seed}")
        random.seed(args.seed)

        print("Creating results directory...")
        os.makedirs(args.results_dir, exist_ok=True)

        print("Loading target code...")
        with open(args.target_snippet_file, "r") as f:
            target_code = f.read()

        print("Target code loaded.")
        print(target_code)

        print("Original generated code:")
        print(target.generate([target_code], PASS_1)[0][0])

        print(f'Trying to generate a code snippet containing "{args.target_substring}"')

        print("Starting attack...")
        attack_single_regex(target, target_code, args.target_substring, repl_ids, args.results_dir)


if __name__ == "__main__":
    attack(parse_args())
