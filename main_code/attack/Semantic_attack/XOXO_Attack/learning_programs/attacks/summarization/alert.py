import argparse
import copy
import random
import json
import os
import time

import torch
import transformers
from tqdm import tqdm
from transformers import RobertaForMaskedLM, RobertaTokenizerFast

from learning_programs.attacks.common import PREPROCESS_SEED
from learning_programs.attacks import common
from learning_programs.attacks.attack_utils import (
    get_identifier_posistions_from_code,
    get_substitues,
    is_valid_substitue,
    map_chromesome,
    select_parents,
    crossover,
    mutate,
    _tokenize,
    set_seed
)
from learning_programs.attacks.parser_utils import (
    get_example,
    get_identifiers,
    is_valid_variable_name,
    remove_comments_and_docstrings,
)
from learning_programs.attacks.print_results import print_results
from learning_programs.attacks.summarization.attack_utils import (
    compute_fitness,
    get_importance_score,
    initialize_target_model,
    get_results,
    get_criterion
)
from learning_programs.attacks.summarization.ours import AttackStats, Result, get_filtered_examples
from learning_programs.datasets.summarization import Example, load_examples

def preprocess(examples: list[Example], tokenizer_mlm: RobertaTokenizerFast, args: argparse.Namespace) -> list[dict[str, list[str]]]:
    set_seed(PREPROCESS_SEED)
    print("Preprocessing the Defect detection dataset for ALERT attack...")

    os.makedirs(args.preprocess_path, exist_ok=True)
    target_path = os.path.join(args.preprocess_path, "preprocess.jsonl")

    if os.path.exists(target_path):
        print(f"Found existing preprocessed dataset at {target_path}.")
        if args.from_scratch:
            print("--from_scratch flag is set. Removing the existing preprocessed dataset.")
            os.remove(target_path)
        else:
            print("Loading the existing preprocessed dataset.")
            with open(target_path, "r") as f:
                return [json.loads(line) for line in f]

    transformers.logging.set_verbosity_error()
    codebert_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm").to(args.device).eval()
    transformers.logging.set_verbosity_info()

    for example in tqdm(examples): 
        identifiers, code_tokens = get_identifiers(remove_comments_and_docstrings(example.code, common.LANGUAGE), common.LANGUAGE)
        processed_code = " ".join(code_tokens)
            
        words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)

        variable_names = []
        for name in identifiers:
            if ' ' in name[0].strip():
                continue
            variable_names.append(name[0])

        sub_words = [tokenizer_mlm.cls_token] + sub_words[:args.block_size - 2] + [tokenizer_mlm.sep_token]
            
        input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])

        word_predictions = codebert_mlm(input_ids_.to(args.device))[0].squeeze()  # seq-len(sub) vocab
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, 60, -1)  # seq-len k

        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]
            
        names_positions_dict = get_identifier_posistions_from_code(words, variable_names)

        variable_substitue_dict = {}
        with torch.no_grad():
            orig_embeddings = codebert_mlm.roberta(input_ids_.to(args.device))[0]

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for tgt_word in names_positions_dict.keys():
            tgt_positions = names_positions_dict[tgt_word] # the positions of tgt_word in code
            if not is_valid_variable_name(tgt_word, lang=common.LANGUAGE):
                # if the extracted name is not valid
                continue   

            all_substitues = []
            for one_pos in tgt_positions:
                if keys[one_pos][0] >= word_predictions.size()[0]:
                    continue
                substitutes = word_predictions[keys[one_pos][0]:keys[one_pos][1]]  # L, k
                word_pred_scores = word_pred_scores_all[keys[one_pos][0]:keys[one_pos][1]]
                    
                orig_word_embed = orig_embeddings[0][keys[one_pos][0]+1:keys[one_pos][1]+1]

                similar_substitutes = []
                similar_word_pred_scores = []
                sims = []
                subwords_leng, nums_candis = substitutes.size()

                for i in range(nums_candis):

                    new_ids_ = copy.deepcopy(input_ids_)
                    new_ids_[0][keys[one_pos][0]+1:keys[one_pos][1]+1] = substitutes[:,i]

                    with torch.no_grad():
                        new_embeddings = codebert_mlm.roberta(new_ids_.to(args.device))[0]
                    new_word_embed = new_embeddings[0][keys[one_pos][0]+1:keys[one_pos][1]+1]

                    sims.append((i, sum(cos(orig_word_embed, new_word_embed))/subwords_leng))
                    
                sims = sorted(sims, key=lambda x: x[1], reverse=True)

                for i in range(int(nums_candis/2)):
                    similar_substitutes.append(substitutes[:,sims[i][0]].reshape(subwords_leng, -1))
                    similar_word_pred_scores.append(word_pred_scores[:,sims[i][0]].reshape(subwords_leng, -1))

                similar_substitutes = torch.cat(similar_substitutes, 1)
                similar_word_pred_scores = torch.cat(similar_word_pred_scores, 1)

                substitutes = get_substitues(similar_substitutes, 
                                            tokenizer_mlm, 
                                            codebert_mlm, 
                                            1,
                                            args.device,
                                            similar_word_pred_scores, 
                                            0)
                all_substitues += substitutes
            all_substitues = set(all_substitues)

            for tmp_substitue in all_substitues:
                if tmp_substitue.strip() in variable_names:
                    continue
                if not is_valid_substitue(tmp_substitue.strip(), tgt_word, common.LANGUAGE):
                    continue
                try:
                    variable_substitue_dict[tgt_word].append(tmp_substitue)
                except:
                    variable_substitue_dict[tgt_word] = [tmp_substitue]

        with open(target_path, 'a') as wf:
            wf.write(json.dumps(variable_substitue_dict)+'\n')

    with open(target_path, "r") as f:
        return [json.loads(line) for line in f]


class Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, tokenizer_mlm, use_bpe, threshold_pred_score) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_mlm = tokenizer_mlm
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score


    def ga_attack(self, example, substituions, initial_replace=None):
        '''
        return
            original program: code
            program length: prog_length
            adversar program: adv_program
            true label: true_label
            original prediction: orig_label
            adversarial prediction: temp_label
            is_attack_success: is_success
            extracted variables: variable_names
            importance score of variables: names_to_importance_score
            number of changed variables: nb_changed_var
            number of changed positions: nb_changed_pos
            substitues for variables: replaced_words
        '''
        time_start = time.time()
        bleus, docstrings, queries = get_results([example.code], example.docstring, self.model_tgt, self.tokenizer_tgt, self.args)
        orig_bleu = bleus[0]
        orig_docstring = docstrings[0]

        identifiers, code_tokens = get_identifiers(example.code, common.LANGUAGE)

        variable_names = list(substituions.keys())

        if len(identifiers) == 0 or len(variable_names) == 0:
            return Result.from_parser_failure(example, time_start)

        names_positions_dict = get_identifier_posistions_from_code(code_tokens, variable_names)

        variable_substitue_dict = {tgt_word : substituions[tgt_word] for tgt_word in names_positions_dict.keys()}

        nb_changed_var = 0
        nb_changed_pos = 0
        fitness_values = []
        current_bleu = orig_bleu
        base_chromesome = {word: word for word in variable_substitue_dict.keys()}
        population = [base_chromesome]
        success_criterion = get_criterion(self.args)

        for tgt_word in variable_substitue_dict.keys():
            if initial_replace is None:
                replace_examples = []
                substitute_list = []
                most_gap = 0.0
                initial_candidate = tgt_word

                for a_substitue in variable_substitue_dict[tgt_word]:
                    substitute_list.append(a_substitue)
                    temp_code = get_example(example.code, tgt_word, a_substitue, common.LANGUAGE)
                    replace_examples.append(temp_code)

                if len(replace_examples) == 0:
                    continue

                bleus, docstrings, queries = get_results(replace_examples, example.docstring, self.model_tgt, self.tokenizer_tgt, self.args, queries)

                _the_best_candidate = -1
                for index, bleu in enumerate(bleus):
                    gap = current_bleu - bleu
                    if gap > most_gap:
                        most_gap = gap
                        _the_best_candidate = index
                if _the_best_candidate == -1:
                    initial_candidate = tgt_word
                else:
                    initial_candidate = substitute_list[_the_best_candidate]
            else:
                initial_candidate = initial_replace[tgt_word]

            temp_chromesome = copy.deepcopy(base_chromesome)
            temp_chromesome[tgt_word] = initial_candidate
            population.append(temp_chromesome)
            temp_fitness, _, queries = compute_fitness(temp_chromesome, self.model_tgt, self.tokenizer_tgt, orig_bleu, example.docstring, example.code, self.args, queries)
            fitness_values.append(temp_fitness)

        cross_probability = 0.7

        max_iter = max(5 * len(population), 5)

        for _ in range(max_iter):
            _temp_mutants = []
            for j in range(10):
                p = random.random()
                chromesome_1, _, chromesome_2, _ = select_parents(population)
                if p < cross_probability:
                    if chromesome_1 == chromesome_2:
                        child_1 = mutate(chromesome_1, variable_substitue_dict)
                        continue
                    child_1, _ = crossover(chromesome_1, chromesome_2)
                    if child_1 == chromesome_1 or child_1 == chromesome_2:
                        child_1 = mutate(chromesome_1, variable_substitue_dict)
                else:
                    child_1 = mutate(chromesome_1, variable_substitue_dict)
                _temp_mutants.append(child_1)
            
            # compute fitness in batch
            feature_list = []
            for mutant in _temp_mutants:
                _temp_code =  map_chromesome(mutant, example.code, common.LANGUAGE)
                feature_list.append(_temp_code)
            if len(feature_list) == 0:
                continue
            bleus, docstrings, queries = get_results(feature_list, example.docstring, self.model_tgt, self.tokenizer_tgt, self.args, queries)
            mutate_fitness_values = []
            for _temp_mutant, bleu, docstring, adv_code in zip(_temp_mutants, bleus, docstrings, feature_list):
                if success_criterion(bleu, orig_bleu):
                    for old_word in _temp_mutant.keys():
                        if old_word == _temp_mutant[old_word]:
                            nb_changed_var += 1
                            nb_changed_pos += len(names_positions_dict[old_word])
                    return Result.from_success(example, queries, orig_bleu, orig_docstring, adv_code, docstring, bleu, nb_changed_var, nb_changed_pos, time_start)
                elif current_bleu > bleu:
                    current_bleu = bleu
                _tmp_fitness = orig_bleu - bleu
                mutate_fitness_values.append(_tmp_fitness)
            
            for index, fitness_value in enumerate(mutate_fitness_values):
                min_value = min(fitness_values)
                if fitness_value > min_value:
                    min_index = fitness_values.index(min_value)
                    population[min_index] = _temp_mutants[index]
                    fitness_values[min_index] = fitness_value

        return Result.from_failure(example, queries, orig_bleu, orig_docstring, current_bleu, time_start)


    def greedy_attack(self, example, substituions):
        '''
        return
            original program: code
            program length: prog_length
            adversar program: adv_program
            true label: true_label
            original prediction: orig_label
            adversarial prediction: temp_label
            is_attack_success: is_success
            extracted variables: variable_names
            importance score of variables: names_to_importance_score
            number of changed variables: nb_changed_var
            number of changed positions: nb_changed_pos
            substitues for variables: replaced_words
        '''
        time_start = time.time()
        bleus, docstrings, queries = get_results([example.code], example.docstring, self.model_tgt, self.tokenizer_tgt, self.args)
        orig_bleu = bleus[0]
        orig_docstring = docstrings[0]

        identifiers, code_tokens = get_identifiers(example.code, common.LANGUAGE)

        variable_names = list(substituions.keys())

        if len(identifiers) == 0 or len(variable_names) == 0:
            return Result.from_parser_failure(example, time_start), None


        importance_score, replace_token_positions, names_positions_dict, queries = get_importance_score(example,
                                                                                               code_tokens,
                                                                                               variable_names,
                                                                                               self.model_tgt,
                                                                                               self.tokenizer_tgt,
                                                                                               self.args,
                                                                                               queries)

        if importance_score is None:
            return Result.from_failure(example, queries, orig_bleu, orig_docstring, orig_bleu, time_start), None

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

            all_substitues = substituions[tgt_word]

            most_gap = 0.0
            candidate = None
            replace_examples = []

            substitute_list = []
            for substitute in all_substitues:
                substitute_list.append(substitute)
                temp_code = get_example(final_code, tgt_word, substitute, common.LANGUAGE)
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
                    return Result.from_success(example, queries, orig_bleu, orig_docstring, replace_example, docstring, bleu, nb_changed_var, nb_changed_pos, time_start), replaced_words
                else:
                    gap = orig_bleu - bleu
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute_list[index]
        
            if most_gap > 0:
                nb_changed_var += 1
                nb_changed_pos += len(names_positions_dict[tgt_word])
                current_bleu = current_bleu - most_gap
                replaced_words[tgt_word] = candidate
                final_code = get_example(final_code, tgt_word, candidate, common.LANGUAGE)
                print("%s ACC! %s => %s (%.5f => %.5f)" % \
                    ('>>', tgt_word, candidate,
                    current_bleu + most_gap,
                    current_bleu), flush=True)
            else:
                replaced_words[tgt_word] = tgt_word

        return Result.from_failure(example, queries, orig_bleu, orig_docstring, current_bleu, time_start), replaced_words

    def attack(self, example, substituions):
        result, replaced_words = self.greedy_attack(example, substituions)
        if result.success is None:
            return result
        if not result.success:
            greedy_time = result.time
            result = self.ga_attack(example, substituions, replaced_words)
            result_dict = result._asdict()
            result_dict["time"] += greedy_time
            result = Result(**result_dict)
        return result


def get_resume_idx(results_path: str) -> int:
    if not os.path.exists(results_path):
        return 0
    results = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results[-1]["idx"] + 1


def parse_args():
    parser = common.ArgumentParser()
    parser.add_argument("--from_scratch", action="store_true", help="Rewrite cached data")
    parser.add_argument("--block_size", default=512, type=int)
    args = parser.parse_args()
    args.use_ga = True
    return args


def attack(args: argparse.Namespace):
    target, tokenizer = initialize_target_model(args)
    print("Loading MLM tokenizer...")
    tokenizer_mlm = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base-mlm")
    print("Loading dataset...")
    examples = load_examples("test")
    print("Preparing attack...")
    substitutes = preprocess(examples, tokenizer_mlm, args)
    print(f"Setting seed {args.seed}...")
    set_seed(args.seed)
    print("Initializing attacker...")
    attacker = Attacker(args, target, tokenizer, tokenizer_mlm, use_bpe=1, threshold_pred_score=0)
    print("Filtering examples...")
    filtered_examples = get_filtered_examples(examples, target, tokenizer, args)
    filtered_substitutes = [s for s, e in zip(substitutes, examples) if e.idx in {e.idx for e in filtered_examples}]
    print("Starting attack...")
    stats = AttackStats()
    with open(args.results_path, "w") as f, tqdm(total=len(filtered_examples)) as pbar:
        for i, (example, substitute) in enumerate(zip(filtered_examples, filtered_substitutes)):
            result = attacker.attack(example, substitute)
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
