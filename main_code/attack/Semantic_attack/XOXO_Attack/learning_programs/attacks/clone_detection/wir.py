import argparse
import copy
import random
import json
import os
import time

import torch
from tqdm import tqdm

from learning_programs.attacks import common
from learning_programs.attacks.attack_utils import (
    set_seed,
    isUID,
    build_vocab
)
from learning_programs.attacks.parser_utils import (
    get_example,
    get_identifiers
)
from learning_programs.attacks.clone_detection.attack_utils import (
    convert_code_to_features,
    get_importance_score,
    initialize_target_model,
    is_correctly_classified,
    CodeDataset
)
from learning_programs.attacks.clone_detection.ours import AttackStats, Result
from learning_programs.attacks.print_results import print_results
from learning_programs.datasets.clone_detection import load_examples

class WIR_Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, _token2idx, _idx2token) -> None:
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.token2idx = _token2idx
        self.idx2token = _idx2token
        self.args = args

    def attack(self, example, raw_example):
        time_start = time.time()
        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[-1].item()
        temp_label = None

        identifiers, code_tokens = get_identifiers(raw_example.code1, common.LANGUAGE)

        variable_names = [identifier[0] for identifier in identifiers]

        if len(identifiers) == 0 or len(variable_names) == 0:
            return Result.from_parser_failure(raw_example, time_start)

        importance_score, replace_token_positions, names_positions_dict = get_importance_score(self.args,
                                                                                               example,
                                                                                               raw_example.code2,
                                                                                               code_tokens,
                                                                                               variable_names,
                                                                                               self.model_tgt,
                                                                                               self.tokenizer_tgt)

        if importance_score is None:
            return Result.from_failure(raw_example, self.model_tgt.query, time_start)

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

        final_code = copy.deepcopy(raw_example.code1)
        nb_changed_var = 0
        nb_changed_pos = 0
        replaced_words = {}

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
                new_feature = convert_code_to_features(temp_code, raw_example.code2, self.tokenizer_tgt, true_label, self.args)
                replace_examples.append(new_feature)
            if len(replace_examples) == 0:
                continue
            new_dataset = CodeDataset(replace_examples, self.args)

            logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            assert (len(logits) == len(substitute_list))

            for index, temp_prob in enumerate(logits):
                temp_label = preds[index]
                if temp_label != orig_label:
                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    candidate = substitute_list[index]
                    replaced_words[tgt_word] = candidate
                    adv_code = get_example(final_code, tgt_word, candidate, common.LANGUAGE) + raw_example.code2
                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                          ('>>', tgt_word, candidate,
                           current_prob,
                           temp_prob[orig_label]), flush=True)
                    return Result.from_success(raw_example, self.model_tgt.query, adv_code, nb_changed_var, nb_changed_pos, time_start)
                else:
                    gap = current_prob - temp_prob[temp_label]
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute_list[index]

            if most_gap > 0:
                nb_changed_var += 1
                nb_changed_pos += len(names_positions_dict[tgt_word])
                current_prob = current_prob - most_gap
                replaced_words[tgt_word] = candidate
                final_code = get_example(final_code, tgt_word, candidate, common.LANGUAGE)
                print("%s ACC! %s => %s (%.5f => %.5f)" % \
                      ('>>', tgt_word, candidate,
                       current_prob + most_gap,
                       current_prob), flush=True)
            else:
                replaced_words[tgt_word] = tgt_word

        return Result.from_failure(raw_example, self.model_tgt.query, time_start)


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
    attacker = WIR_Attacker(args, target, tokenizer, token2id, id2token)
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
