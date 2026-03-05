import argparse
import copy
import json
import os
import random
import time

import torch
from tqdm import tqdm

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
from learning_programs.attacks.clone_detection.attack_utils import (
    convert_code_to_features,
    initialize_target_model,
    is_correctly_classified,
    CodeDataset
)
from learning_programs.attacks.clone_detection.ours import AttackStats, Result
from learning_programs.attacks.print_results import print_results
from learning_programs.datasets.clone_detection import load_examples

class MHM_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, _token2idx, _idx2token) -> None:
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args

    def attack(self, example, raw_example, _n_candi=30, _max_iter=50):
        self.time_start = time.time()
        identifiers, code_tokens = get_identifiers(raw_example.code1, common.LANGUAGE)
        variable_names = filter_valid_variable_names(identifiers, common.LANGUAGE)
        true_label = example[-1].item()

        if len(identifiers) == 0 or len(variable_names) == 0:
            return Result.from_parser_failure(raw_example, self.time_start)

        positions = get_identifier_posistions_from_code(code_tokens, variable_names)

        original_vars = set(variable_names)
        original_positions = copy.deepcopy(positions)
        replaced_vars = set()
        code = raw_example.code1
        for _ in range(1, 1 + _max_iter):
            success, code, old_uid, new_uid = self.__replaceUID(code, raw_example.code2, true_label, variable_names, _n_candi)
            if old_uid in original_vars:
                replaced_vars.add(old_uid)
            positions[new_uid] = positions.pop(old_uid)
            variable_names = list(positions.keys())
            if success:
                num_changed_vars = len(replaced_vars)
                num_changed_pos = sum(len(original_positions[candi_uid]) for candi_uid in replaced_vars)
                return Result.from_success(raw_example, self.model_tgt.query, code + raw_example.code2, num_changed_vars, num_changed_pos, self.time_start)
        return Result.from_failure(raw_example, self.model_tgt.query, self.time_start)

    def __replaceUID(self, code, code2, label, variable_names, _n_candi=30):
        selected_uid = random.choice(variable_names)
        candi_uids = []
        candi_codes = []
        candi_examples = []
        for c in random.sample(self.idx2token, _n_candi):  
            if isUID(c):
                candi_uids.append(c)
                candi_codes.append(get_example(copy.deepcopy(code), selected_uid, c, common.LANGUAGE))
                candi_examples.append(convert_code_to_features(candi_codes[-1], code2, self.tokenizer_tgt, label, self.args))

        new_dataset = CodeDataset(candi_examples, self.args)
        _, pred = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)

        for pred, candi_uid, candi_code in zip(pred, candi_uids, candi_codes):
            if pred != label:
                return True, candi_code, selected_uid, candi_uid
        return False, candi_code, selected_uid, candi_uid


def attack(args: argparse.Namespace):
    # This implementation forks the hf tokenizer, so the fast tokenizer is automatically disabled. This env variable is set only to silence the warning,
    # it forces no behavior that the tokenizer would not do by default. However, we still prefer to use the fast tokenizer for the hf tokenizer to improve
    # its performance and give it a fair comparison with the other attacks.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    target, tokenizer = initialize_target_model(args)
    print("Loading dataset...")
    examples = load_examples("test")
    dataset = CodeDataset.from_split("test", tokenizer, args)
    print(f"Setting seed {args.seed}...")
    set_seed(args.seed)
    print("Initializing attacker...")
    code_tokens = [get_identifiers(example.code1, common.LANGUAGE)[1] for example in examples]
    id2token, token2id = build_vocab(code_tokens, 5000)
    attacker = MHM_Attacker(args, target, tokenizer, token2id, id2token)
    print("Starting attack...")
    stats = AttackStats()
    with open(args.results_path, "w") as f, tqdm(total=len(dataset)) as pbar:
        for i, (example, raw_example) in enumerate(zip(dataset, examples)):
            if not is_correctly_classified(target, example, args.eval_batch_size):
                pbar.update()
                continue
            result = attacker.attack(example, raw_example)
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
    with torch.no_grad():
        attack(common.ArgumentParser().parse_args())
