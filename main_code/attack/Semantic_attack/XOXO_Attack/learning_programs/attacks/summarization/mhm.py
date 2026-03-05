import argparse
import copy
import json
import random
import time

import transformers
from tqdm import tqdm
from transformers import RobertaTokenizerFast, RobertaForMaskedLM

from learning_programs.attacks import common
from learning_programs.attacks.attack_utils import (
    get_identifier_posistions_from_code,
    set_seed,
    isUID,
    build_vocab
)
from learning_programs.attacks.parser_utils import (
    get_example,
    get_identifiers,
    filter_valid_variable_names
)
from learning_programs.attacks.print_results import print_results
from learning_programs.attacks.summarization.attack_utils import (
    initialize_target_model,
    get_results,
    get_criterion,
)
from learning_programs.attacks.summarization.ours import AttackStats, Result, get_filtered_examples
from learning_programs.datasets.summarization import load_examples

class MHM_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, model_mlm, tokenizer_mlm, _token2idx, _idx2token) -> None:
        self.model_tgt = model_tgt
        self.model_mlm = model_mlm
        self.tokenizer_tgt = tokenizer_tgt
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args
        self.tokenizer_mlm = tokenizer_mlm

    def attack(self, example, _n_candi=25, _max_iter=10):
        time_start = time.time()
        identifiers, code_tokens = get_identifiers(example.code, common.LANGUAGE)
        variable_names = filter_valid_variable_names(identifiers, common.LANGUAGE)

        if len(identifiers) == 0 or len(variable_names) == 0:
            return Result.from_parser_failure(example, time_start)

        positions = get_identifier_posistions_from_code(code_tokens, variable_names)

        bleus, docstrings, queries = get_results([example.code], example.docstring, self.model_tgt, self.tokenizer_tgt, self.args, 0)
        orig_bleu = bleus[0]
        orig_docstring = docstrings[0]
        self.best_bleu = orig_bleu

        original_vars = set(variable_names)
        original_positions = copy.deepcopy(positions)
        replaced_vars = set()
        code = example.code
        for _ in range(1, 1 + _max_iter):
            success, code, docstring, old_uid, new_uid, queries = self.__replaceUID(example, code, orig_bleu, variable_names, queries, _n_candi)
            if old_uid in original_vars:
                replaced_vars.add(old_uid)
            positions[new_uid] = positions.pop(old_uid)
            variable_names = list(positions.keys())
            if success:
                num_changed_vars = len(replaced_vars)
                num_changed_pos = sum(len(original_positions[candi_uid]) for candi_uid in replaced_vars)
                return Result.from_success(example, queries, orig_bleu, orig_docstring, code, docstring, self.best_bleu, num_changed_vars, num_changed_pos, time_start)
        return Result.from_failure(example, queries, orig_bleu, orig_docstring, self.best_bleu, time_start)

    def __replaceUID(self, example, code, orig_bleu, variable_names, queries_so_far, _n_candi):
        success_criterion = get_criterion(self.args)
        selected_uid = random.choice(variable_names)
        candi_uids = []
        candi_codes = []
        for c in random.sample(self.idx2token, _n_candi):
            if isUID(c):
                candi_uids.append(c)
                candi_codes.append(get_example(copy.deepcopy(code), selected_uid, c, common.LANGUAGE))

        bleus, docstrings, queries = get_results(candi_codes, example.docstring, self.model_tgt, self.tokenizer_tgt, self.args, queries_so_far)

        for bleu, docstring, candi_uid, candi_code in zip(bleus, docstrings, candi_uids, candi_codes):
            if bleu < self.best_bleu:
                self.best_bleu = bleu
            if success_criterion(bleu, orig_bleu):
                return True, candi_code, docstring, selected_uid, candi_uid, queries
        return False, candi_code, docstring, selected_uid, candi_uid, queries


def attack(args: argparse.Namespace):
    target, tokenizer = initialize_target_model(args)
    print("Loading MLM tokenizer...")
    tokenizer_mlm = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base-mlm")
    print("Loading dataset...")
    examples = load_examples("test")
    print("Loading surrogate model...")
    transformers.logging.set_verbosity_error()
    codebert_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm").to(args.device).eval()
    transformers.logging.set_verbosity_info()
    print(f"Setting seed {args.seed}...")
    set_seed(args.seed)
    print("Initializing attacker...")
    code_tokens = [get_identifiers(example.code, common.LANGUAGE)[1] for example in examples]
    id2token, token2id = build_vocab(code_tokens, 5000)
    attacker = MHM_Attacker(args, target, tokenizer, codebert_mlm, tokenizer_mlm, token2id, id2token)
    print("Filtering examples...")
    filtered_examples = get_filtered_examples(examples, target, tokenizer, args)
    print("Starting attack...")
    stats = AttackStats()
    with open(args.results_path, "w") as f, tqdm(total=len(filtered_examples)) as pbar:
        for i, example in enumerate(filtered_examples):
            result = attacker.attack(example)
            stats.update(result)
            pbar.update()
            pbar.set_description(str(stats))
            if result.success is None:
                pbar.write(f"Example {i}: Failed to find any identifiers")
            else:
                pbar.write(f"Example {i}: {'Success' if result.success else 'Failure'}")
            f.write(json.dumps(result._asdict()) + "\n")
    print_results(args.results_path)


if __name__ == "__main__":
    attack(common.ArgumentParser().parse_args())
