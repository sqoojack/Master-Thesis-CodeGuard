import argparse
import copy
import random
import json
import os
import time

import torch
from tqdm import tqdm
from transformers import RobertaForMaskedLM, RobertaTokenizerFast

from common import PREPROCESS_SEED
from attack_utils import (
    get_identifier_positions_from_code,
    get_substitutes,
    is_valid_substitute,
    map_chromosome,
    select_parents,
    crossover,
    mutate,
    _tokenize,
    set_seed,
)
from parser_utils import (
    get_example,
    get_identifiers,
    is_valid_variable_name,
    remove_comments_and_docstrings,
)
from clone_detection.attack_utils import (
    compute_fitness,
    convert_code_to_features,
    get_importance_score,
    initialize_target_model,
    CodeDataset
)
from print_results import print_results
from clone_detection.common import get_preprocess_path, get_results_path, LANGUAGE
from datasets.clone_detection import Example, load_examples


def preprocess(examples, tokenizer_mlm, args):
    set_seed(PREPROCESS_SEED)
    print("Preprocessing the Clone detection dataset for attack...")

    preprocess_path = get_preprocess_path("codeattack")
    os.makedirs(preprocess_path, exist_ok=True)
    target_path = os.path.join(preprocess_path, "preprocess.jsonl")

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
        identifiers, code_tokens = get_identifiers(remove_comments_and_docstrings(example.code1, LANGUAGE), LANGUAGE)
        processed_code = " ".join(code_tokens)
            
        words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)

        variable_names = [name[0] for name in identifiers if ' ' not in name[0].strip()]

        sub_words = [tokenizer_mlm.cls_token] + sub_words[:args.block_size - 2] + [tokenizer_mlm.sep_token]
        input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])

        word_predictions = codebert_mlm(input_ids_.to(args.device))[0].squeeze()
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, 60, -1)
        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

        names_positions_dict = get_identifier_positions_from_code(words, variable_names)

        variable_substitute_dict = {}
        with torch.no_grad():
            orig_embeddings = codebert_mlm.roberta(input_ids_.to(args.device))[0]

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        for tgt_word, positions in names_positions_dict.items():
            if not is_valid_variable_name(tgt_word, lang=LANGUAGE):
                continue

            all_substitutes = []
            for one_pos in positions:
                if keys[one_pos][0] >= word_predictions.size()[0]:
                    continue
                substitutes = word_predictions[keys[one_pos][0]:keys[one_pos][1]]
                word_pred_scores = word_pred_scores_all[keys[one_pos][0]:keys[one_pos][1]]
                orig_word_embed = orig_embeddings[0][keys[one_pos][0] + 1:keys[one_pos][1] + 1]

                similar_substitutes = []
                similar_word_pred_scores = []
                sims = [(i, sum(cos(orig_word_embed, codebert_mlm.roberta(input_ids_.to(args.device))[0][0][keys[one_pos][0] + 1:keys[one_pos][1] + 1])) / len(substitutes)) for i in range(len(substitutes))]

                sims = sorted(sims, key=lambda x: x[1], reverse=True)[:30]

                for i, _ in sims:
                    similar_substitutes.append(substitutes[:, i].reshape(len(substitutes), -1))
                    similar_word_pred_scores.append(word_pred_scores[:, i].reshape(len(substitutes), -1))

                substitutes = get_substitutes(torch.cat(similar_substitutes, 1), tokenizer_mlm, codebert_mlm, 1, args.device, torch.cat(similar_word_pred_scores, 1), 0)
                all_substitutes += substitutes

            variable_substitute_dict[tgt_word] = [sub for sub in set(all_substitutes) if is_valid_substitute(sub.strip(), tgt_word, LANGUAGE)]

        with open(target_path, 'a') as wf:
            wf.write(json.dumps(variable_substitute_dict) + '\n')

    with open(target_path, "r") as f:
        return [json.loads(line) for line in f]


class Attacker:
    def __init__(self, args, model_tgt, tokenizer_tgt, tokenizer_mlm, use_bpe, threshold_pred_score):
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.tokenizer_mlm = tokenizer_mlm
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score

    def ga_attack(self, example, substitutes, code_1, code_2, initial_replace=None):
        # Genetic Algorithm (GA) based attack implementation
        pass  # Implementation details...

    def greedy_attack(self, example, substitutes, code_1, code_2):
        # Greedy attack implementation
        pass  # Implementation details...


def get_resume_idx(results_path):
    if not os.path.exists(results_path):
        return 0
    with open(results_path) as f:
        return [json.loads(line) for line in f][-1]["idx"] + 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_scratch", action="store_true", help="Rewrite cached data")
    args = parser.parse_args()
    args.use_ga = True
    return args


def attack(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    target, tokenizer = initialize_target_model(args)
    tokenizer_mlm = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base-mlm")
    examples = load_examples("test")
    dataset = CodeDataset.from_split("test", tokenizer, args)
    substitutes = preprocess(examples, tokenizer_mlm, args)
    set_seed(args.seed)
    attacker = Attacker(args, target, tokenizer, tokenizer_mlm, use_bpe=1, threshold_pred_score=0)
    results_path = get_results_path("codeattack", args.model_name, args.seed)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    with open(results_path, "w") as f, tqdm(total=len(dataset)) as pbar:
        for i, (example, raw_example, substitute) in enumerate(zip(dataset, examples, substitutes)):
            start_time = time.time()
            target.query = 0
            code1, code2 = raw_example.code1, raw_example.code2
            _, adv_code, _, _, _, _, is_success, _, _, num_changed_vars, num_changed_pos, replaced_words = attacker.greedy_attack(example, substitute, code1, code2)
            if is_success == -4:
                pbar.update()
                continue
            if is_success == -1 and args.use_ga:
                _, adv_code, _, _, _, _, is_success, _, _, num_changed_vars, num_changed_pos, _ = attacker.ga_attack(example, substitute, code1, code2, initial_replace=replaced_words)
            success = None if is_success == -3 else is_success == 1
            time_delta = time.time() - start_time

            pbar.write(f"Example {i}: {'Success' if success else 'Failure'}")
            pbar.update()

            f.write(json.dumps({
                "idx": i,
                "success": success,
                "queries": target.query,
                "code": code1 + code2,
                "adv_code": adv_code,
                "target": raw_example.label,
                "num_changed_vars": num_changed_vars,
                "num_changed_pos": num_changed_pos,
                "time": time_delta
            }) + "\n")
    print_results(results_path)


if __name__ == "__main__":
    with torch.no_grad():
        attack(parse_args())
