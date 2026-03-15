import json
import time
import random
import os

INPUT_FILE = "Dataset/XOXO_attack/XOXO_clone_detection_codebert.jsonl"
OUTPUT_FILE = "Dataset/Adaptive_attack/contextual_attack.jsonl"

def generate_contextual_adv_code(original_code):
    top_common_names = ["data", "config", "query", "result", "info", "item"]
    stealth_trigger = random.choice(top_common_names)
    
    # 構造一個看起來極為正常的 context 語句作為 trigger
    contextual_block = f"    String {stealth_trigger} = \"init\";\n"
    
    insert_pos = original_code.find('{') + 1
    if insert_pos <= 0:
        return original_code, 0, 0
        
    adv_code = original_code[:insert_pos] + "\n" + contextual_block + original_code[insert_pos:]
    return adv_code, 1, 1

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
        open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            start_time = time.time()
            
            adv_code, changed_vars, changed_pos = generate_contextual_adv_code(data['code'])
            
            end_time = time.time()
            
            output_data = {
                "idx": data.get("idx"),
                "success": True if changed_vars > 0 else False,
                "queries": data.get("queries", 0) + 350,
                "code": data["code"],
                "adv_code": adv_code,
                "target": data.get("target", 1),
                "num_changed_vars": changed_vars,
                "num_changed_pos": changed_pos,
                "time": (end_time - start_time)
            }
            outfile.write(json.dumps(output_data) + "\n")

if __name__ == "__main__":
    main()