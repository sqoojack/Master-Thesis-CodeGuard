import argparse
import copy
import heapq
import json
import random
import time
from collections import defaultdict

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
from learning_programs.attacks.print_results import print_results
from learning_programs.attacks.summarization.attack_utils import (
    initialize_target_model,
    get_results,
    get_criterion
)
from learning_programs.attacks.summarization.ours import AttackStats, Result, get_filtered_examples
from learning_programs.datasets.summarization import load_examples

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


def _get_var_importance_score_by_uncertainty(example, words_list: list, variable_names: list, tgt_model, tokenizer, args, queries_so_far=0):
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    if len(positions) == 0:
        return None, None, queries_so_far

    masked_token_list, masked_var_list = get_replaced_var_code_with_meaningless_char(words_list, positions)

    new_codes = [" ".join(tokens) for tokens in [words_list] + masked_token_list]
    bleus, _, queries = get_results(new_codes, example.docstring, tgt_model, tokenizer, args, queries_so_far)

    orig_bleu = bleus[0]

    var_importance_score_by_variance = defaultdict(list)
    for bleu, var in zip(bleus[1:], masked_var_list):
        var_importance_score_by_variance[var].append(orig_bleu - bleu)
    
    for var in var_importance_score_by_variance:
        var_importance_score_by_variance[var] = np.var(var_importance_score_by_variance[var])

    return var_importance_score_by_variance, positions, queries + queries_so_far


class RnnsAttacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, model_mlm, tokenizer_mlm, use_bpe, threshold_pred_score) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.model_mlm = model_mlm
        self.tokenizer_mlm = tokenizer_mlm
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score
        self.variable_emb, self.variable_name = self._get_variable_info()

    def _get_variable_info(self):
        variable_embs = []
        
        codebert_mlm = self.model_mlm 
        tokenizer_mlm = self.tokenizer_mlm

        codes = [ex.code for ex in load_examples("train") + load_examples("valid") + load_examples("test")]

        variables = []
        for code in tqdm(codes):
            identifiers, _ = get_identifiers(code, common.LANGUAGE)
            variables.extend(filter_valid_variable_names(identifiers))

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
    
    def attack(self, example):
        time_start = time.time()
        bleus, docstrings, queries = get_results([example.code], example.docstring, self.model_tgt, self.tokenizer_tgt, self.args)
        orig_bleu = bleus[0]
        orig_docstring = docstrings[0]

        identifiers, code_tokens = get_identifiers(example.code, common.LANGUAGE)

        variable_names = filter_valid_variable_names(identifiers, common.LANGUAGE)

        if len(identifiers) == 0 or len(variable_names) == 0:
            return Result.from_parser_failure(example, time_start)

        substituions = {}

        names_to_importance_score, names_positions_dict, queries = _get_var_importance_score_by_uncertainty(example, code_tokens, variable_names, self.model_tgt, self.tokenizer_tgt, self.args, queries)

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)

        final_code = copy.deepcopy(example.code)
        nb_changed_var = 0  
        nb_changed_pos = 0
        replaced_words = {}
        success_criterion = get_criterion(self.args)
        current_bleu = orig_bleu

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
                    replace_examples.append(temp_code)
                if len(replace_examples) == 0:
                    break

                bleus, docstrings, queries = get_results(replace_examples, example.docstring, self.model_tgt, self.tokenizer_tgt, self.args, queries)
    
                used_candidate.extend(all_substitues)
                
                golden_bleu_decrease_track = {}
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
                        if gap > 0:
                            golden_bleu_decrease_track[substitute_list[index]] = gap

                if len(golden_bleu_decrease_track) > 0:
                    cur_iter_track = {}
                    sorted_golden_bleu_decrease_track = sorted(golden_bleu_decrease_track.items(), key=lambda x: x[1],
                                                               reverse=True)
                    (candidate, most_gap) = sorted_golden_bleu_decrease_track[0]
                    cur_iter_track["candidate"] = list(map(lambda x: x[0], sorted_golden_bleu_decrease_track[1:]))

                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    current_bleu = orig_bleu - most_gap
                    if candidate not in valid_variable_names or best_candidate not in valid_variable_names:
                        replaced_words[tgt_word] = best_candidate
                        final_code = get_example(final_code, tgt_word, best_candidate, common.LANGUAGE)
                        break

                    candidate_index = valid_variable_names.index(candidate)
                    best_candidate_index = valid_variable_names.index(best_candidate)

                    bleu_delt_emb = variable_embs[candidate_index] - variable_embs[best_candidate_index]
                    if momentum is None:
                        momentum = bleu_delt_emb   
                    momentum = (1 - self.args.a) * momentum + self.args.a * bleu_delt_emb
                    
                    if self.args.rnns_type == "RNNS-Delta":
                        virtual_emb = variable_embs[candidate_index] + bleu_delt_emb
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
                               current_bleu + most_gap,
                               current_bleu), flush=True)

                    break

                loop_time += 1
                if loop_time >= self.args.iters:
                    if best_candidate != tgt_word:
                        replaced_words[tgt_word] = best_candidate
                        final_code = get_example(final_code, tgt_word, best_candidate, common.LANGUAGE)
                        print("%s ACC! %s => %s (%.5f => %.5f)" % \
                              ('>>', tgt_word, best_candidate,
                               current_bleu + most_gap,
                               current_bleu), flush=True)
                    break

        return Result.from_failure(example, queries, orig_bleu, orig_docstring, current_bleu, time_start)
    

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
    attacker = RnnsAttacker(args, target, tokenizer, codebert_mlm, tokenizer_mlm, use_bpe=1, threshold_pred_score=0)
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
    args = common.ArgumentParser().parse_args()
    # Setting the hyperparameters for the attacker as per the paper #https://github.com/18682922316/RNNS-for-code-attack/tree/main/CodeBert/authorship
    args.substitutes_size = 10
    args.rnns_type = "RNNS-Smooth"
    args.a = 0.2
    args.iters = 6
    args.max_distance = 0.15
    args.max_length_diff = 3

    with torch.no_grad():
        attack(args)
