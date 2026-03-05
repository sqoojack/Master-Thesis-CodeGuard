import argparse
import copy
import heapq
import json
import os
import random
import time

import numpy as np
import torch
import transformers
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaForMaskedLM, RobertaTokenizerFast

from learning_programs.attacks import common
from learning_programs.attacks.attack_utils import (
    get_identifier_posistions_from_code,
    set_seed
)
from learning_programs.attacks.parser_utils import (
    get_example,
    get_identifiers,
    filter_valid_variable_names
)
from learning_programs.attacks.defect_detection.attack_utils import (
    initialize_target_model,
    is_correctly_classified,
    convert_code_to_features,
    CodeDataset
)
from learning_programs.attacks.defect_detection.ours import AttackStats, Result
from learning_programs.attacks.print_results import print_results
from learning_programs.datasets.defect_detection import load_examples

def get_replaced_var_code_with_meaningless_char(tokens: list, positions: dict):
    masked_token_list = []
    masked_var_list = []
    for variable_name in positions.keys():
        for tmp_var in ["a","b","c","de", "fg","hi","jkl","mno","pqr","stuv","wxyz","abcde","fghig","klmno","pqrst","uvwxyz"]:
            tmp_tokens = copy.deepcopy(tokens)
            for pos in positions[variable_name]:
                tmp_tokens[pos] = tmp_var

            masked_token_list.append(tmp_tokens)
            masked_var_list.append(variable_name)

    return masked_token_list, masked_var_list


def _get_var_importance_score_by_uncertainty(args, example, words_list: list, variable_names: list, tgt_model, tokenizer):
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    if len(positions) == 0:
        return None, None

    new_example = []

    masked_token_list, masked_var_list = get_replaced_var_code_with_meaningless_char(words_list, positions)

    for tokens in [words_list] + masked_token_list:
        new_code = ' '.join(tokens)
        new_feature = convert_code_to_features(new_code, tokenizer, example[-1].item(), args)
        new_example.append(new_feature)
    new_dataset = CodeDataset(new_example, args)

    logits, preds = tgt_model.get_results(new_dataset, args.eval_batch_size)
    orig_label = preds[0]

    var_pos_delt_prob_disp = {}
    var_neg_delt_prob_disp = {}
    var_importance_score_by_variance = {}
    for prob, var in zip(logits[1:], masked_var_list):
        if var in var_pos_delt_prob_disp:
            var_pos_delt_prob_disp[var].append(prob[orig_label])
            var_neg_delt_prob_disp[var].append(1 - prob[orig_label])
        else:
            var_pos_delt_prob_disp[var] = [prob[orig_label]]
            var_neg_delt_prob_disp[var] = [1 - prob[orig_label]]

    for var in var_pos_delt_prob_disp:
        VarP = (np.var(var_pos_delt_prob_disp[var]) + np.var(var_neg_delt_prob_disp[var])) / 2
        var_importance_score_by_variance[var] = VarP

    return var_importance_score_by_variance, positions


class RnnsAttacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, use_bpe, threshold_pred_score):
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score
        self.variable_emb, self.variable_name = self._get_variable_info()

    def _get_variable_info(self):
        variables = []
        variable_embs = []
        
        transformers.logging.set_verbosity_error()
        tokenizer_mlm = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base-mlm")
        codebert_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm").to(self.args.device).eval()
        transformers.logging.set_verbosity_info()

        codes = [ex.code for ex in load_examples("train") + load_examples("valid") + load_examples("test")]

        variables = []

        for code in codes:
            identifiers, _ = get_identifiers(code, common.LANGUAGE)
            variables.extend(filter_valid_variable_names(identifiers, lang=common.LANGUAGE))

        variables = sorted(set(variables))

        for var in tqdm(variables):
            sub_words = tokenizer_mlm.tokenize(var)
            sub_words = [tokenizer_mlm.cls_token] + sub_words[:self.args.block_size - 2] + [tokenizer_mlm.sep_token]
            input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])
            with torch.no_grad():
                orig_embeddings = codebert_mlm.roberta(input_ids_.to(self.args.device))[0][0][1:-1]
                mean_embedding = torch.mean(orig_embeddings, dim=0, keepdim=True).cpu().detach().numpy()[0]
                variable_embs.append(mean_embedding)

        assert len(variable_embs) == len(variables)
        return variable_embs, variables
    
    def attack(self, example, raw_example):
        time_start = time.time()
        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[-1].item()
        temp_label = None

        identifiers, code_tokens = get_identifiers(raw_example.code, common.LANGUAGE)

        variable_names = filter_valid_variable_names(identifiers, lang=common.LANGUAGE)

        substituions = {}

        if len(identifiers) == 0 or len(variable_names) == 0:
            return Result.from_parser_failure(raw_example, time_start)

        names_to_importance_score, names_positions_dict = _get_var_importance_score_by_uncertainty(self.args, example, code_tokens, variable_names, self.model_tgt, self.tokenizer_tgt)

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)

        final_code = copy.deepcopy(raw_example.code)
        nb_changed_var = 0  
        nb_changed_pos = 0
        replaced_words = {}

        for name_and_score in sorted_list_of_names:
            used_candidate = list(replaced_words.values())
            tgt_word = name_and_score[0]
            tgt_word_len = len(tgt_word)

            tgt_index = self.variable_name.index(tgt_word)
            distances = 1 - cosine_similarity(np.array(self.variable_emb),np.array([self.variable_emb[tgt_index]]))
            variable_index_list = [i for i,  distance in enumerate(distances) if distance < self.args.max_distance and len(self.variable_name[i])<= tgt_word_len + self.args.max_length_diff]
            variable_embs = [ self.variable_emb[index] for index in variable_index_list]
            valid_variable_names = [ self.variable_name[index] for index in variable_index_list]
            
            index_list = [i for i in range(0, len(valid_variable_names))]
            random.shuffle(index_list)
            inds_1 = index_list[:self.args.substitutes_size]
            inds_2 = index_list[self.args.substitutes_size:3*self.args.substitutes_size]
            all_substitues = [valid_variable_names[ind] for ind in inds_1]
            substituions[tgt_word] =  [valid_variable_names[ind] for ind in inds_2]

            candidate = None
            new_substitutes = []
            for sub in all_substitues:
                if sub not in used_candidate:
                    new_substitutes.append(sub)
                       
            best_candidate = tgt_word
            loop_time = 0
            momentum = None
            track = []
            while True:
                substitute_list = []
                replace_examples = []
                most_gap = 0.0
                for substitute in new_substitutes:
                    substitute_list.append(substitute)
                    temp_code = get_example(final_code, tgt_word, substitute, common.LANGUAGE)
                    new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt, true_label, self.args)
                    replace_examples.append(new_feature)
                if len(replace_examples) == 0:
                    break
                new_dataset = CodeDataset(replace_examples, self.args)
    
                logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
                assert (len(logits) == len(substitute_list))
                used_candidate.extend(all_substitues)

                golden_prob_decrease_track = {}
                for index, temp_prob in enumerate(logits):
                    temp_label = preds[index]
                    if temp_label != orig_label:
                        nb_changed_var += 1
                        nb_changed_pos += len(names_positions_dict[tgt_word])
                        candidate = substitute_list[index]
                        replaced_words[tgt_word] = candidate
                        adv_code = get_example(final_code, tgt_word, candidate, common.LANGUAGE)
                        print("%s SUC! %s => %s (%.5f => %.5f)" % \
                              ('>>', tgt_word, candidate,
                               current_prob,
                               temp_prob[orig_label]), flush=True)
                        return Result.from_success(raw_example, self.model_tgt.query, adv_code, nb_changed_var, nb_changed_pos, time_start)
                    else:
                        gap = current_prob - temp_prob[temp_label]
                        if gap > 0:
                            golden_prob_decrease_track[substitute_list[index]] = gap

                if len(golden_prob_decrease_track) > 0:
                    cur_iter_track = {}
                    sorted_golden_prob_decrease_track = sorted(golden_prob_decrease_track.items(), key=lambda x: x[1],
                                                               reverse=True)
                    (candidate, most_gap) = sorted_golden_prob_decrease_track[0]
                    cur_iter_track["candidate"] = list(map(lambda x: x[0], sorted_golden_prob_decrease_track[1:]))

                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    current_prob = current_prob - most_gap
                    if candidate not in valid_variable_names or best_candidate not in valid_variable_names:
                        replaced_words[tgt_word] = best_candidate
                        final_code = get_example(final_code, tgt_word, best_candidate, common.LANGUAGE)
                        break

                    candidate_index = valid_variable_names.index(candidate)
                    best_candidate_index = valid_variable_names.index(best_candidate)

                    prob_delt_emb = variable_embs[candidate_index] - variable_embs[best_candidate_index]
                    if momentum is None:
                        momentum = prob_delt_emb   
                    momentum = (1 - self.args.a) * momentum + self.args.a * prob_delt_emb

                    if self.args.rnns_type == "RNNS-Delta":
                        virtual_emb = variable_embs[candidate_index] + prob_delt_emb
                    elif self.args.rnns_type == "RNNS-Smooth":
                        virtual_emb = variable_embs[candidate_index] + momentum
                    elif self.args.rnns_type == "RNNS-Raw":
                        virtual_emb = variable_embs[candidate_index]
                    else:
                        pass

                    similarity = cosine_similarity(np.array(variable_embs),
                                                   np.array([virtual_emb]))

                    inds = heapq.nlargest(len(similarity), range(len(similarity)), similarity.__getitem__)
                    new_substitutes.clear()
                    if len(replaced_words) > 0:
                        used_candidate.extend(list(replaced_words.values()))

                    for ind in inds:
                        temp_var = valid_variable_names[ind]
                        if temp_var not in used_candidate:
                            new_substitutes.append(temp_var)
                            if temp_var in substituions[tgt_word]:
                                substituions[tgt_word].remove(temp_var)
                            used_candidate.append(temp_var)
                            if len(new_substitutes) >= self.args.substitutes_size:
                                break

                    best_candidate = candidate
                    cur_iter_track["best_candidate"] = best_candidate
                    track.append(cur_iter_track)

                else:
                    if best_candidate != tgt_word:
                        replaced_words[tgt_word] = best_candidate
                        final_code = get_example(final_code, tgt_word, best_candidate, common.LANGUAGE)
                        print("%s ACC! %s => %s (%.5f => %.5f)" % \
                              ('>>', tgt_word, best_candidate,
                               current_prob + most_gap,
                               current_prob), flush=True)
                    break

                loop_time += 1
                if loop_time >= self.args.iters:
                    if best_candidate != tgt_word:
                        replaced_words[tgt_word] = best_candidate
                        final_code = get_example(final_code, tgt_word, best_candidate, common.LANGUAGE)
                        print("%s ACC! %s => %s (%.5f => %.5f)" % \
                              ('>>', tgt_word, best_candidate,
                               current_prob + most_gap,
                               current_prob), flush=True)
                    break

        return Result.from_failure(raw_example, self.model_tgt.query, time_start)


def parse_args() -> argparse.Namespace:
    args = common.ArgumentParser().parse_args()
    # Setting the hyperparameters for the attacker as per the paper #https://github.com/18682922316/RNNS-for-code-attack/tree/main/CodeBert/authorship
    args.substitutes_size = 60
    args.rnns_type = "RNNS-Smooth"
    args.a = 0.2
    args.iters = 6
    args.max_distance = 0.15
    args.max_length_diff = 3
    return args


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
    attacker = RnnsAttacker(args, target, tokenizer, use_bpe=1, threshold_pred_score=0)
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
        attack(parse_args())
