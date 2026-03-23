# python main_code/attack/Adaptive_attack/contextual.py
import json
import time
import re
import os
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_FILE = "Dataset/merged_all/tiny_merged_dataset.jsonl"
OUTPUT_FILE = "Dataset/Adaptive_attack/contextual_attack.jsonl"
MODEL_ID = "Salesforce/codegen-350M-mono" # 必須與防禦端模型一致以確保白箱/灰箱效果

# 基因演算法超參數
POP_SIZE = 10       # 每個 Trigger 的候選族群大小
GENERATIONS = 8     # 演化代數
MUTATION_RATE = 0.4 # 突變機率

class SurrogateEvaluator:
    """代理評估器：用來計算適應度函數 (Fitness Function)"""
    def __init__(self, model_id, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(self.device)
        self.model.eval()

    def get_token_losses(self, input_ids):
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return losses.cpu().numpy()

    def evaluate_surprise(self, prefix, trigger):
        """計算目標函數：Surprise Score = max(0, L_context - L_prior)"""
        # 1. 計算 L_prior (單獨存在時的 Loss)
        inputs_trigger = self.tokenizer(trigger, return_tensors="pt").to(self.device)
        if inputs_trigger["input_ids"].shape[1] <= 1: return 0.0
        prior_losses = self.get_token_losses(inputs_trigger["input_ids"])
        prior = np.mean(prior_losses) if len(prior_losses) > 0 else 0.0

        # 2. 計算 L_context (在給定 Prefix 上下文時的 Loss)
        text = prefix + "\n    " + trigger
        inputs_full = self.tokenizer(text, return_tensors="pt").to(self.device)
        full_losses = self.get_token_losses(inputs_full["input_ids"])

        # 僅擷取 Trigger 部分的 Loss 進行比對
        trigger_len = inputs_trigger["input_ids"].shape[1]
        if len(full_losses) >= trigger_len:
            context_losses = full_losses[-trigger_len:]
            context = np.mean(context_losses)
        else:
            context = np.mean(full_losses)

        # 回傳適應度 (越低越好)
        return max(0.0, context - prior)

def genetic_algorithm_optimize(evaluator, trigger):
    """基因演算法主體：優化出能讓模型預測出該 Trigger 的最佳 Context"""
    # 基因庫：常見 C 語言關鍵字、無害變數名、註解符號
    vocab = ['data', 'config', 'query', 'result', 'info', 'item', 'init', 'setup', 
             'buffer', 'stream', 'int', 'void', 'char', 'struct', 'if', 'return', 
             'true', 'false', 'NULL', 'TODO:', 'NOTE:', 'Fix', 'handling', 'for', 
             '=', ';', '()', '/*', '*/', '//', trigger]

    # 初始化族群 (Population)
    population = [
        f"// Initialize {trigger} state",
        f"/* {trigger} configuration */",
        f"bool _{trigger}_ready = true;",
        f"struct {trigger}_t {trigger}_obj;"
    ]
    while len(population) < POP_SIZE:
        random_words = " ".join(random.choices(vocab, k=random.randint(3, 8)))
        population.append(f"/* {random_words} */")

    best_prefix = population[0]
    best_fitness = float('inf')

    # 世代演化
    for gen in range(GENERATIONS):
        fitness_scores = []
        for indiv in population:
            # 計算 Surprise Score 作為 Fitness (越小越好)
            score = evaluator.evaluate_surprise(indiv, trigger)
            fitness_scores.append((score, indiv))

        fitness_scores.sort(key=lambda x: x[0])

        if fitness_scores[0][0] < best_fitness:
            best_fitness = fitness_scores[0][0]
            best_prefix = fitness_scores[0][1]

        # 若已達完美中性化 (Surprise == 0)，提早結束演化
        if best_fitness <= 0.001:
            break

        # 菁英保留 (Elitism)
        new_population = [fitness_scores[0][1], fitness_scores[1][1]]

        # 繁衍下一代
        while len(new_population) < POP_SIZE:
            parent1 = random.choice(fitness_scores[:POP_SIZE//2])[1]
            parent2 = random.choice(fitness_scores[:POP_SIZE//2])[1]

            # 交配 (Crossover)
            split1 = len(parent1) // 2
            split2 = len(parent2) // 2
            child = parent1[:split1] + parent2[split2:]

            # 突變 (Mutation)
            if random.random() < MUTATION_RATE:
                mut_type = random.choice(['insert', 'replace', 'wrap_comment'])
                if mut_type == 'insert':
                    child += " " + random.choice(vocab)
                elif mut_type == 'replace' and len(child.split()) > 0:
                    words = child.split()
                    words[random.randint(0, len(words)-1)] = random.choice(vocab)
                    child = " ".join(words)
                elif mut_type == 'wrap_comment':
                    child = f"/* {child} */"

            new_population.append(child)
        population = new_population

    # 確保最終產出的是合法的 C 程式碼區塊
    final_output = f"    {best_prefix}\n"
    return final_output, best_fitness

def extract_triggers(original_code, adv_code):
    orig_words = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', original_code))
    adv_words = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', adv_code))
    triggers = list(adv_words - orig_words)
    c_keywords = {'int', 'char', 'void', 'float', 'double', 'long', 'short', 'unsigned', 'signed', 
                  'struct', 'union', 'enum', 'static', 'const', 'volatile', 'register', 'auto', 
                  'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default', 'break', 'continue', 
                  'return', 'goto', 'sizeof', 'typedef', 'true', 'false', 'NULL', 'bool', 
                  'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t'}
    return [t for t in triggers if t not in c_keywords]

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print(f"[-] Loading Surrogate Model ({MODEL_ID}) for Genetic Algorithm...")
    evaluator = SurrogateEvaluator(MODEL_ID)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        
    print(f"[-] Starting Genetic Algorithm Optimization on {len(lines)} samples...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for line in tqdm(lines, ncols=80, desc="Optimizing Context"):
            data = json.loads(line.strip())
            start_time = time.time()
            
            original_code = data.get("code", "")
            base_prompt = data.get("adv_code", "")
            if not base_prompt: base_prompt = original_code
            
            triggers = extract_triggers(original_code, base_prompt)
            
            if triggers:
                # 針對最主要的 Trigger 進行基因演算法優化
                target_trigger = triggers[0]
                optimal_context, final_surprise = genetic_algorithm_optimize(evaluator, target_trigger)
                
                # 將優化出的完美 Context 注入至頂層或第一個大括號後
                insert_pos = base_prompt.find('{') + 1
                if insert_pos > 0:
                    adv_code = base_prompt[:insert_pos] + "\n" + optimal_context + base_prompt[insert_pos:]
                else:
                    adv_code = optimal_context + base_prompt
                    
                changed_vars = 1
            else:
                adv_code = base_prompt
                changed_vars = 0
            
            end_time = time.time()
            
            output_data = {
                "idx": data.get("idx"),
                "success": True if changed_vars > 0 else False,
                "queries": data.get("queries", 0) + (POP_SIZE * GENERATIONS), # 記錄優化成本
                "code": original_code,
                "adv_code": adv_code,
                "target": data.get("target", 1),
                "num_changed_vars": changed_vars,
                "num_changed_pos": 1 if changed_vars > 0 else 0,
                "time": (end_time - start_time) + data.get("time", 0)
            }
            outfile.write(json.dumps(output_data) + "\n")
            
    print(f"write to {OUTPUT_FILE} successfully")

if __name__ == "__main__":
    main()