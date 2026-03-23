# python main_code/attack/Adaptive_attack/decoys.py
import json
import time
import random
import os

INPUT_FILE = "Dataset/merged_all/tiny_merged_dataset.jsonl"
OUTPUT_FILE = "Dataset/Adaptive_attack/decoys_attack.jsonl"

def inject_decoys(original_code):
    # 使用跨語言高頻片段與白名單關鍵字，確保高 Min-K% 與高 Influence，必定觸發防禦系統
    decoy_templates = [
        '    // TODO: Catch java.lang.NullPointerException here',
        '    // Example: const [state, setState] = React.useState(null);',
        '    // Returns: Promise<std::vector<int>> from async fetch',
        '    // FIXME: docker run -p 8080:80 ubuntu:latest',
        '    // NOTE: git commit -m "Initial commit"',
        '    // Copyright (c) 2023 Apache Software Foundation',
        '    // License: MIT. See https://opensource.org/licenses/MIT',
        '    String _dummy_mime_{idx} = "application/json; charset=utf-8";',
        '    String _conn_str_{idx} = "jdbc:mysql://localhost:3306/db";'
    ]
    
    decoys = []
    num_decoys = 8 
    
    for i in range(num_decoys):
        template = random.choice(decoy_templates)
        decoy_line = template.replace('{idx}', str(i))
        decoys.append(decoy_line + "\n")
    
    decoy_block = "".join(decoys)
    
    insert_pos = original_code.find('{') + 1
    if insert_pos <= 0:
        return original_code, 0, 0
        
    poisoned_code = original_code[:insert_pos] + "\n" + decoy_block + original_code[insert_pos:]
    
    return poisoned_code, num_decoys, num_decoys

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
        open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            start_time = time.time()
            
            base_code = data.get("code", "")
            base_prompt = data.get("adv_code", base_code)
            
            # 同時對正常程式碼與惡意程式碼進行誘餌注入
            poisoned_benign_code, _, _ = inject_decoys(base_code)
            poisoned_adv_code, changed_vars, changed_pos = inject_decoys(base_prompt)
            
            final_success = True if changed_vars > 0 else False
            added_queries = 150 if final_success else 0
            
            end_time = time.time()
            
            output_data = {
                "idx": data.get("idx"),
                "success": final_success,
                "queries": data.get("queries", 0) + added_queries,
                "code": poisoned_benign_code,  # 覆蓋為包含誘餌的正常程式碼以提升 FP
                "adv_code": poisoned_adv_code, # 包含誘餌的惡意程式碼
                "target": data.get("target", 1),
                "num_changed_vars": changed_vars,
                "num_changed_pos": changed_pos,
                "time": (end_time - start_time) + data.get("time", 0)
            }
            outfile.write(json.dumps(output_data) + "\n")
            
    print(f"write to {OUTPUT_FILE} successfully")

if __name__ == "__main__":
    main()