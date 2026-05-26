"""
python main_code/evaluate/Inference_XOXO.py \
    --model_name_or_path microsoft/codebert-base \
    --model_weight_path main_code/attack/XOXO_Attack/learning_programs/datasets/CodeXGLUE/Code-Code/Defect-detection/code/saved_models/2019024262/microsoft/codebert-base/checkpoint-best-acc/model.bin \
    --attack_path result/sanitized_data/CodeGuard_sanitized_XOXO_0.015_0.10.jsonl \
    -L3_b 0.015  \
    -L3_t 0.10  \
    --seed 68
    
python main_code/evaluate/Inference_XOXO.py \
    --model_name_or_path microsoft/graphcodebert-base \
    --model_weight_path main_code/attack/XOXO_Attack/learning_programs/datasets/CodeXGLUE/Code-Code/Defect-detection/code/saved_models/2019024262/microsoft/graphcodebert-base/checkpoint-best-acc/model.bin \
    --attack_path result/sanitized_data/CodeGuard_sanitized_XOXO_0.015_0.10.jsonl \
    -L3_b 0.015  \
    -L3_t 0.10  \
    --seed 68
    
python main_code/evaluate/Inference_XOXO.py \
    --model_name_or_path microsoft/codebert-base \
    --model_weight_path main_code/attack/XOXO_Attack/learning_programs/datasets/CodeXGLUE/Code-Code/Defect-detection/code/saved_models/2019024262/microsoft/codebert-base/checkpoint-best-acc/model.bin \
    --attack_path result/sanitized_data/KillBadCode_XOXO_clean.jsonl \
    --seed 68
    
python main_code/evaluate/Inference_XOXO.py \
    --model_name_or_path microsoft/graphcodebert-base \
    --model_weight_path main_code/attack/XOXO_Attack/learning_programs/datasets/CodeXGLUE/Code-Code/Defect-detection/code/saved_models/2019024262/microsoft/graphcodebert-base/checkpoint-best-acc/model.bin \
    --attack_path result/sanitized_data/KillBadCode_XOXO_clean.jsonl \
    --seed 68
"""

import json
import torch
import argparse
import torch.nn as nn
import random
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers import logging as transformers_logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ============================================================
# 模型定義
# ============================================================
class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Model(nn.Module):
    def __init__(self, encoder, config, pad_token_id):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.classifier = RobertaClassificationHead(config)
        self.pad_token_id = pad_token_id

    def forward(self, input_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(self.pad_token_id))[0]
        logits = self.classifier(outputs)
        prob = torch.sigmoid(logits)
        return prob

# ============================================================
# 設定與推論
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Defense Utility (ACC/F1) and Robustness (ASR)")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/codebert-base")
    parser.add_argument("--model_weight_path", type=str, required=True)
    parser.add_argument("--attack_path", type=str, required=True, help="Path to repaired_results.jsonl")
    parser.add_argument("-L3_b", "--l3_base_influence", type=float, default=0.025, help="Base strict threshold for variables.")
    parser.add_argument("-L3_t", "--l3_surprise_tolerance", type=float, default=0.10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--block_size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def get_prediction(model, tokenizer, code, device, block_size):
    if not code or code.strip() == "": 
        # 空代碼通常視為安全 (0) 或保持原判，這裡假設模型會判為 0
        return 0
        
    code_tokens = tokenizer.tokenize(code)[:block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    input_ids = torch.tensor([source_ids]).to(device)
    
    with torch.no_grad():
        prob = model(input_ids)
        # Prob > 0.5 = Vulnerable (1), Prob <= 0.5 = Safe (0)
        prediction = 1 if prob.item() > 0.5 else 0
    return prediction

def main():
    transformers_logging.set_verbosity_error()
    args = parse_args()
    set_seed(args.seed)
    
    # 1. Load Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = 1
    encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config)
    model = Model(encoder, config, tokenizer.pad_token_id)
    
    # 2. Load Weights
    print(f"[-] Loading Weights: {args.model_weight_path}")
    state_dict = torch.load(args.model_weight_path, map_location="cpu")
    if "model_state_dict" in state_dict: state_dict = state_dict["model_state_dict"]
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("encoder.roberta."): 
            new_state_dict[k.replace("encoder.roberta.", "encoder.")] = v
        elif k.startswith("encoder.classifier."):
            new_state_dict[k.replace("encoder.classifier.", "classifier.")] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(args.device)
    model.eval()

    # 3. Evaluate
    print(f"[-] Evaluating Utility & Robustness: {args.attack_path}")
    
    y_true = []
    y_pred_adv = []      # 攻擊後的預測 (未防禦)
    y_pred_repaired = [] # 防禦後的預測
    
    total_samples = 0
    successful_attacks_orig = 0
    successful_attacks_repair = 0

    with open(args.attack_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Inference", ncols=100):
        try: entry = json.loads(line)
        except: continue
        
        # 獲取真實標籤
        label = int(entry.get("target", entry.get("label", -1)))
        
        # 為了計算 Utility，我們只關注原本是有毒的樣本 (Label=1)
        # 因為 defense output file 只有對 adv_code 進行修復
        if label != 1: 
            continue 

        total_samples += 1
        y_true.append(label)
        
        # 1. 攻擊代碼 (Adversarial Code)
        adv_code = entry.get("adv_code", "")
        pred_a = get_prediction(model, tokenizer, adv_code, args.device, args.block_size)
        y_pred_adv.append(pred_a)
        
        if pred_a == 0: successful_attacks_orig += 1 # 攻擊成功 (騙成0)
        
        # 2. 修復代碼 (Repaired Code)
        # 如果該行沒有 repaired_code (例如防禦沒觸發)，則使用原始 adv_code
        repaired_code = entry.get("repaired_code", adv_code)
        pred_r = get_prediction(model, tokenizer, repaired_code, args.device, args.block_size)
        y_pred_repaired.append(pred_r)
        
        if pred_r == 0: successful_attacks_repair += 1 # 攻擊依然成功 (防禦失敗)

    # 4. Metrics Calculation
    if total_samples == 0:
        print("No adversarial samples found.")
        return

    # ASR (Attack Success Rate) - Lower is better
    asr_orig = (successful_attacks_orig / total_samples) * 100
    asr_repair = (successful_attacks_repair / total_samples) * 100
    
    acc_orig = accuracy_score(y_true, y_pred_adv) * 100
    acc_repair = accuracy_score(y_true, y_pred_repaired) * 100

    print("\n" + "="*60)
    print(f"Evaluation Report (Total Samples: {total_samples})")
    print("-" * 60)
    print(f"{'Metric':<20} | {'No Defense':<15} | {'With Defense':<15} | {'Improvement':<10}")
    print("-" * 60)
    print(f"{'ASR (Lower better)':<20} | {asr_orig:<14.2f}% | {asr_repair:<14.2f}% | {asr_orig - asr_repair:+.2f}%")
    print(f"{'ACC (Utility)':<20} | {acc_orig:<14.2f}% | {acc_repair:<14.2f}% | {acc_repair - acc_orig:+.2f}%")
    print("="*60)
    
    output_dir = "result/evaluation"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_data = {
        "config": {
            "model_type": args.model_name_or_path,
            "attack_source": args.attack_path,
            "l3_base_influence": args.l3_base_influence,
            "l3_surprise_tolerance": args.l3_surprise_tolerance
        },
        "metrics": {
            "ASR": {
                "before_defense": round(asr_orig, 4),
                "after_defense": round(asr_repair, 4),
                "improvement": round(asr_orig - asr_repair, 4)
            },
            "ACC": {
                "before_defense": round(acc_orig, 4),
                "after_defense": round(acc_repair, 4),
                "improvement": round(acc_repair - acc_orig, 4)
            },
        }

    }

    output_path = os.path.join(output_dir, "ASR_XOXO_baseline.json")
    
    # 讀取現有資料
    existing_data = []
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as jf:
                existing_data = json.load(jf)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except (json.JSONDecodeError, Exception):
            existing_data = []
    import datetime
    results_data["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    existing_data.append(results_data)

    # 寫回檔案
    with open(output_path, 'w', encoding='utf-8') as jf:
        json.dump(existing_data, jf, indent=4, ensure_ascii=False)
    
    print(f"\n[+] Results appended to: {output_path}")

if __name__ == "__main__":
    main()