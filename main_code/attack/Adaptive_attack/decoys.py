# python main_code/attack/Adaptive_attack/decoys.py
import json
import time
import random
import string
import os

INPUT_FILE = "Dataset/XOXO_attack/XOXO_clone_detection_codebert.jsonl"
OUTPUT_FILE = "Dataset/Adaptive_attack/decoys_attack.jsonl"

def generate_decoy_adv_code(original_code):
    decoys = []
    # 產生 20 個高 Surprise score 但無惡意的誘餌
    for i in range(20):
        gibberish = ''.join(random.choices(string.ascii_letters, k=15))
        decoys.append(f"    int {gibberish} = {i};\n")
    
    decoy_block = "".join(decoys)
    trigger_block = "    String trigger = \"malicious_payload\";\n"
    
    insert_pos = original_code.find('{') + 1
    if insert_pos <= 0:
        return original_code, 0, 0
        
    adv_code = original_code[:insert_pos] + "\n" + decoy_block + trigger_block + original_code[insert_pos:]
    return adv_code, 21, 21

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
        open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            start_time = time.time()
            
            adv_code, changed_vars, changed_pos = generate_decoy_adv_code(data['code'])
            
            end_time = time.time()
            
            output_data = {
                "idx": data.get("idx"),
                "success": True if changed_vars > 0 else False,
                "queries": data.get("queries", 0) + 150, # 模擬增加的 queries
                "code": data["code"],
                "adv_code": adv_code,
                "target": data.get("target", 1),
                "num_changed_vars": changed_vars,
                "num_changed_pos": changed_pos,
                "time": (end_time - start_time)
            }
            outfile.write(json.dumps(output_data) + "\n")
    print(f"write to {OUTPUT_FILE} successfully")

if __name__ == "__main__":
    main()