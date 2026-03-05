import argparse
import copy
import random
import json
import time

import torch
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from learning_programs.attacks import common
from learning_programs.attacks.attack_utils import (
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
    get_importance_score,
    initialize_target_model,
    get_results,
    get_criterion
)
from learning_programs.attacks.summarization.ours import AttackStats, Result, get_filtered_examples
from learning_programs.datasets.summarization import load_examples

class WIR_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, tokenizer_mlm, _token2idx, _idx2token) -> None:
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args
        self.tokenizer_mlm = tokenizer_mlm

    def attack(self, example):
        time_start = time.time()
        bleus, docstrings, queries = get_results([example.code], example.docstring, self.model_tgt, self.tokenizer_tgt, self.args)
        orig_bleu = bleus[0]
        orig_docstring = docstrings[0]

        identifiers, code_tokens = get_identifiers(example.code, common.LANGUAGE)

        variable_names = filter_valid_variable_names(identifiers, common.LANGUAGE)

        if len(identifiers) == 0 or len(variable_names) == 0:
            return Result.from_parser_failure(example, time_start)

        importance_score, replace_token_positions, names_positions_dict, queries = get_importance_score(
                                                                                               example,
                                                                                               code_tokens,
                                                                                               variable_names,
                                                                                               self.model_tgt,
                                                                                               self.tokenizer_tgt,
                                                                                               self.args,
                                                                                               queries)

        if importance_score is None:
            return Result.from_failure(example, queries, orig_bleu, orig_docstring, orig_bleu, time_start)

        token_pos_to_score_pos = {}

        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i
        names_to_importance_score = {}

        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                total_score += importance_score[token_pos_to_score_pos[token_pos]]

            names_to_importance_score[name] = total_score

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)

        final_code = copy.deepcopy(example.code)
        nb_changed_var = 0
        nb_changed_pos = 0
        replaced_words = {}
        success_criterion = get_criterion(self.args)
        current_bleu = orig_bleu

        for name_and_score in sorted_list_of_names:
            tgt_word = name_and_score[0]

            all_substitues = []
            num = 0
            while num < 30:
                tmp_var = random.choice(self.idx2token)
                if isUID(tmp_var):
                    all_substitues.append(tmp_var)
                    num += 1

            most_gap = 0.0
            candidate = None
            replace_examples = []

            substitute_list = []
            for substitute in all_substitues:
                substitute_list.append(substitute)
                temp_code = get_example(final_code, tgt_word, substitute, common.LANGUAGE)
                temp_code = " ".join(temp_code.split())
                replace_examples.append(temp_code)
            if len(replace_examples) == 0:
                continue

            bleus, docstrings, queries = get_results(replace_examples, example.docstring, self.model_tgt, self.tokenizer_tgt, self.args, queries)

            for index, (bleu, docstring, replace_example) in enumerate(zip(bleus, docstrings, replace_examples)):
                if success_criterion(bleu, orig_bleu):
                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    candidate = substitute_list[index]
                    replaced_words[tgt_word] = candidate
                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                          ('>>', tgt_word, candidate,
                           current_bleu, bleu), flush=True)
                    return Result.from_success(example, queries, orig_bleu, orig_docstring, replace_example, docstring, bleu, nb_changed_var, nb_changed_pos, time_start)
                else:
                    gap = orig_bleu - bleu
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute_list[index]

            if most_gap > 0:
                nb_changed_var += 1
                nb_changed_pos += len(names_positions_dict[tgt_word])
                current_bleu = orig_bleu - most_gap
                replaced_words[tgt_word] = candidate
                final_code = get_example(final_code, tgt_word, candidate, common.LANGUAGE)
                print("%s ACC! %s => %s (%.5f => %.5f)" % \
                      ('>>', tgt_word, candidate,
                       current_bleu + most_gap,
                       current_bleu), flush=True)
            else:
                replaced_words[tgt_word] = tgt_word

        return Result.from_failure(example, queries, orig_bleu, orig_docstring, current_bleu, time_start)


def attack(args: argparse.Namespace):
    target, tokenizer = initialize_target_model(args)
    print("Loading MLM tokenizer...")
    tokenizer_mlm = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base-mlm")
    print("Loading dataset...")
    examples = load_examples("test")
    print(f"Setting seed {args.seed}...")
    set_seed(args.seed)
    print("Initializing attacker...")
    code_tokens = [get_identifiers(example.code, common.LANGUAGE)[1] for example in examples]
    id2token, token2id = build_vocab(code_tokens, 5000)
    attacker = WIR_Attacker(args, target, tokenizer, tokenizer_mlm, token2id, id2token)
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
    with torch.no_grad():
        attack(common.ArgumentParser().parse_args())
