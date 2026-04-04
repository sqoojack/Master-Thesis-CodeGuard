import os
import re
import sys

sys.path.append('../../../')
sys.path.append('../code')
sys.path.append('../../../python_parser')
retval = os.getcwd()

import json
import pickle
import logging
import argparse
import warnings
import torch
import time
import random
import numpy as np
from model import Model
from utils import set_seed, get_code_tokens, build_vocab
from run import TextDataset
from run_parser import get_identifiers
from ITGenAttacker import ITGen_Attacker
from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")  # Only report warning

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)
}

logger = logging.getLogger(__name__)

def get_code_pairs(args):
    file_path = args.eval_data_file
    postfix=file_path.split('/')[-1].split('.txt')[0]
    folder = '/'.join(file_path.split('/')[:-1])
    code_pairs_file_path = os.path.join(folder, '{}_cached_{}.pkl'.format(args.model_name, postfix))
    with open(code_pairs_file_path, 'rb') as f:
        code_pairs = pickle.load(f)
    return code_pairs


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--csv_store_path", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=3407, #42
                        help="random seed for initialization")
    
    parser.add_argument("--model_name", default="codebert", type=str,
                        help="The name of model which will be attacked.")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    args.device = device

    # Set seed
    set_seed(args.seed)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    model = Model(model, config, tokenizer, args)

    checkpoint_prefix = 'checkpoint-best-f1/%s_model.bin' % args.model_name
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir), False)
    model.to(args.device)
    logger.info("reload model from {}".format(output_dir))

    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    codebert_mlm.to(args.device)

    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file, pool = None)
    ## Load code pairs
    source_codes = get_code_pairs(args)

    subs_path = "../../../dataset/preprocess/test_subs_clone.jsonl"
    generated_substitutions = []
    with open(subs_path) as f:
        for line in f:
            js = json.loads(line.strip())
            generated_substitutions.append(js["substitutes"])
    assert len(source_codes) == len(eval_dataset) == len(generated_substitutions)

    code_tokens = []
    for index, code in enumerate(source_codes):
        try:
            # Check if code[2] is a string and not empty
            if isinstance(code[2], str) and code[2].strip():
                identifiers_result = get_identifiers(code[2], "java")
                
                if identifiers_result is not None and len(identifiers_result) > 1:
                    code_tokens.append(identifiers_result[1])
            else:
                # If it's an int or empty string, skip it
                continue
                
        except Exception as e:
            print(f"Skipping index {index} due to parser error: {e}")
            continue

    id2token, _ = build_vocab(code_tokens, 5000)

    success_attack = 0
    total_cnt = 0
    attacker = ITGen_Attacker(args, model, tokenizer)
    start_time = time.time()
    query_times = 0
    uid_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    
    output_folder = os.path.dirname(args.csv_store_path)
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"Total number of indices: {len(eval_dataset)}")
    # with open(args.csv_store_path, "w") as wf:
    # 1. 將 "w" 改為 "a" 以進行追加寫入
    with open(args.csv_store_path, "a") as wf:
        for index, example in enumerate(eval_dataset):
            # if index >= 200:
            #     break
            
            # 2. 跳過已經跑過的 Index 0 ~ 200
            if index < 1001:
                continue
                
            # 3. 設定執行到 Index 1000 就停止
            if index > 1300:
                break
            
            tmp_save = {"Index":None,"Original Code":None,"Adversarial Code":None,"Program Length":None,"Identifier Num":None,"Replaced Identifiers":None,"Query Times":None,"Time Cost":None,"Type":None}
            print("Index: ", index)
            code_pair = source_codes[index]
            logits, preds = model.get_results([example], args.eval_batch_size)
            orig_label = preds[0]
            true_label = example[1].item()
            if not orig_label == true_label:
                continue

            substitutes = generated_substitutions[index]
            first_code = code_pair[2]

            orig_code_tokens = get_code_tokens(first_code)
            identifiers = list(substitutes.keys())
            if len(identifiers) == 0:
                continue
            
            num_required = len(identifiers) * 50
            
            if not id2token:
                # 如果詞表為空，無法進行替換，跳過此樣本
                print(f"Index {idx}: id2token is empty, skipping.")
                continue

            if len(id2token) >= num_required:
                selected_tmp_sub = random.sample(id2token, num_required)
            else:
                # 如果可用標識符不足，使用有放回抽樣 (random.choices)
                selected_tmp_sub = random.choices(id2token, k=num_required)
                
            sublists = [selected_tmp_sub[i:i+50] for i in range(0, len(selected_tmp_sub), 50)]
            tmp_sub = []
            for sub in sublists:
                tmp = []
                for s in sub:
                    if bool(uid_pattern.match(s)) and first_code.find(s) == -1:
                        tmp.append(s)
                tmp_sub.append(tmp)
            selected_sub = dict(zip(substitutes.keys(), tmp_sub))

            total_cnt += 1

            example_start_time = time.time()
            try:
                adv_code, is_success, replaced_words = attacker.itgen_attack(example, selected_sub, code_pair, query_times, logits, example_start_time)
            except IndexError as e:
                print(f"Skipping Index {index} due to Parser IndexError: {e}")
                adv_code, is_success, replaced_words = None, 0, None
            except Exception as e:
                print(f"Skipping Index {index} due to unexpected error: {e}")
                adv_code, is_success, replaced_words = None, 0, None

            attack_type = "itgen"

            example_end_time = (time.time() - example_start_time) / 60
            print("Example time cost: ", round(example_end_time, 2), "min")
            print("ALL examples time cost: ", round((time.time() - start_time) / 60, 2), "min")
            print("Query times in this attack: ", model.query - query_times)
            replace_info = ''
            if replaced_words is not None:
                for key in replaced_words.keys():
                    replace_info += key + ':' + replaced_words[key] + ','

            if is_success == 1:
                success_attack += 1
                tmp_save["Index"] = index
                tmp_save["Original Code"] = code_pair[2]
                tmp_save["Adversarial Code"] = adv_code
                tmp_save["Program Length"] = len(orig_code_tokens)
                tmp_save["Identifier Num"] = len(identifiers)
                tmp_save["Replaced Identifiers"] = replace_info
                tmp_save["Query Times"] = model.query - query_times
                tmp_save["Time Cost"] = example_end_time
                tmp_save["Type"] = attack_type
            else:
                tmp_save["Index"] = index
                tmp_save["Program Length"] = len(orig_code_tokens)
                tmp_save["Identifier Num"] = len(identifiers)
                tmp_save["Query Times"] = model.query - query_times
                tmp_save["Time Cost"] = example_end_time
                tmp_save["Type"] = "0"
            query_times = model.query
            wf.write(json.dumps(tmp_save)+'\n')
            print("Success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))
    print("Final success rate: {}/{} = {}".format(success_attack, total_cnt, 1.0 * success_attack / total_cnt))


if __name__ == '__main__':
    main()