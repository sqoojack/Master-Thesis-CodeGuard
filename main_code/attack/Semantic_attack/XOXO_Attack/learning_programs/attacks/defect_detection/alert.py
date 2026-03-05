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
from learning_programs.attacks.defect_detection.attack_utils import (
    compute_fitness,
    convert_code_to_features,
    get_importance_score,
    initialize_target_model,
    is_correctly_classified,
    CodeDataset
)
from learning_programs.attacks.defect_detection.ours import AttackStats, Result
from learning_programs.attacks.print_results import print_results
from learning_programs.datasets.defect_detection import Example, load_examples

def preprocess(examples: list[Example], args: argparse.Namespace) -> list[dict[str, list[str]]]:
    set_seed(common.PREPROCESS_SEED)
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
    tokenizer_mlm = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base-mlm")
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
    def __init__(self, args, model_tgt, tokenizer_tgt, use_bpe, threshold_pred_score) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score


    def ga_attack(self, example, raw_example, substituions, initial_replace=None):
        time_start = time.time()
        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[-1].item()
        adv_code = ''
        temp_label = None

        identifiers, code_tokens = get_identifiers(raw_example.code, common.LANGUAGE)

        variable_names = list(substituions.keys())

        if len(identifiers) == 0 or len(variable_names) == 0:
            return Result.from_parser_failure(raw_example, time_start)

        names_positions_dict = get_identifier_posistions_from_code(code_tokens, variable_names)

        nb_changed_var = 0
        nb_changed_pos = 0

        variable_substitue_dict = {}

        for tgt_word in names_positions_dict.keys():
            variable_substitue_dict[tgt_word] = substituions[tgt_word]

        if len(variable_substitue_dict) == 0:
            return Result.from_failure(raw_example, self.model_tgt.query, time_start)

        fitness_values = []
        base_chromesome = {word: word for word in variable_substitue_dict.keys()}
        population = [base_chromesome]
        for tgt_word in variable_substitue_dict.keys():
            if initial_replace is None:
                replace_examples = []
                substitute_list = []
                current_prob = max(orig_prob)
                most_gap = 0.0
                initial_candidate = tgt_word

                for a_substitue in variable_substitue_dict[tgt_word]:
                    substitute_list.append(a_substitue)
                    temp_code = get_example(raw_example.code, tgt_word, a_substitue, common.LANGUAGE)
                    new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt, true_label, self.args)
                    replace_examples.append(new_feature)

                if len(replace_examples) == 0:
                    continue
                new_dataset = CodeDataset(replace_examples, self.args)
                logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)

                _the_best_candidate = -1
                for index, temp_prob in enumerate(logits):
                    temp_label = preds[index]
                    gap = current_prob - temp_prob[temp_label]
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
            temp_fitness, temp_label = compute_fitness(temp_chromesome, self.model_tgt, self.tokenizer_tgt, max(orig_prob), orig_label, true_label, raw_example.code, names_positions_dict, self.args)
            fitness_values.append(temp_fitness)

        cross_probability = 0.7

        max_iter = max(5 * len(population), 10)

        for _ in range(max_iter):
            _temp_mutants = []
            for j in range(64):
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

            feature_list = []
            for mutant in _temp_mutants:
                _temp_code =  map_chromesome(mutant, raw_example.code, common.LANGUAGE)
                _tmp_feature = convert_code_to_features(_temp_code, self.tokenizer_tgt, true_label, self.args)
                feature_list.append(_tmp_feature)
            if len(feature_list) == 0:
                continue
            new_dataset = CodeDataset(feature_list, self.args)
            mutate_logits, mutate_preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            mutate_fitness_values = []
            for index, logits in enumerate(mutate_logits):
                if mutate_preds[index] != orig_label:
                    adv_code = map_chromesome(_temp_mutants[index], raw_example.code, common.LANGUAGE)
                    for old_word in _temp_mutants[index].keys():
                        if old_word == _temp_mutants[index][old_word]:
                            nb_changed_var += 1
                            nb_changed_pos += len(names_positions_dict[old_word])

                    return Result.from_success(raw_example, self.model_tgt.query, adv_code, nb_changed_var, nb_changed_pos, time_start)
                _tmp_fitness = max(orig_prob) - logits[orig_label]
                mutate_fitness_values.append(_tmp_fitness)

            for index, fitness_value in enumerate(mutate_fitness_values):
                min_value = min(fitness_values)
                if fitness_value > min_value:
                    min_index = fitness_values.index(min_value)
                    population[min_index] = _temp_mutants[index]
                    fitness_values[min_index] = fitness_value

        return Result.from_failure(raw_example, self.model_tgt.query, time_start)


    def greedy_attack(self, example, raw_example, substituions):
        time_start = time.time()
        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[-1].item()
        adv_code = ''
        temp_label = None

        identifiers, code_tokens = get_identifiers(raw_example.code, common.LANGUAGE)

        variable_names = list(substituions.keys())

        if len(identifiers) == 0 or len(variable_names) == 0:
            return Result.from_parser_failure(raw_example, time_start), {}

        importance_score, replace_token_positions, names_positions_dict = get_importance_score(self.args,
                                                                                               example,
                                                                                               code_tokens,
                                                                                               variable_names,
                                                                                               self.model_tgt,
                                                                                               self.tokenizer_tgt)

        if importance_score is None:
            return Result.from_failure(raw_example, self.model_tgt.query, time_start), {}

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

        final_code = copy.deepcopy(raw_example.code)
        nb_changed_var = 0
        nb_changed_pos = 0
        replaced_words = {}

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

                new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt, true_label, self.args)
                replace_examples.append(new_feature)
            if len(replace_examples) == 0:
                continue
            new_dataset = CodeDataset(replace_examples, self.args)
            logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            assert(len(logits) == len(substitute_list))


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
                    return Result.from_success(raw_example, self.model_tgt.query, adv_code, nb_changed_var, nb_changed_pos, time_start), replaced_words
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

            adv_code = final_code

        return Result.from_failure(raw_example, self.model_tgt.query, time_start), replaced_words

    def attack(self, example, raw_example, substituions):
        result, replaced_words = self.greedy_attack(example, raw_example, substituions)
        if result.success is None:
            return result
        if not result.success:
            greedy_time = result.time
            result = self.ga_attack(example, raw_example, substituions, replaced_words)
            result_dict = result._asdict()
            result_dict["time"] += greedy_time
            result = Result(**result_dict)
        return result


def parse_args() -> argparse.Namespace:
    parser = common.ArgumentParser()
    parser.add_argument("--from_scratch", action="store_true", help="Rewrite cached data")
    args = parser.parse_args()
    args.use_ga = True
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
    print("Preparing attack...")
    substitutes = preprocess(examples, args)
    print(f"Setting seed {args.seed}...")
    set_seed(args.seed)
    print("Initializing attacker...")
    attacker = Attacker(args, target, tokenizer, use_bpe=1, threshold_pred_score=0)
    print("Starting attack...")
    stats = AttackStats()
    with open(args.results_path, "w") as f, tqdm(total=len(dataset)) as pbar:
        for i, (example, raw_example, substitute) in enumerate(zip(dataset, examples, substitutes)):
            if not is_correctly_classified(target, example, args.eval_batch_size):
                pbar.update()
                continue
            result = attacker.attack(example, raw_example, substitute)
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
