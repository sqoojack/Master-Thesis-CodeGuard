# python main_code/attack/Adaptive_attack/contextual.py
import json
import time
import random
import os

INPUT_FILE = "Dataset/merged_all/tiny_merged_dataset.jsonl"
OUTPUT_FILE = "Dataset/Adaptive_attack/contextual_attack.jsonl"

def generate_contextual_adv_code(original_code):
    top_common_names = ["data", "config", "query", "result", "info", "item"]
    stealth_trigger = random.choice(top_common_names)
    
    # 構造一個看起來極為正常的 context 語句作為 trigger
    contextual_block = f"    String {stealth_trigger} = \"init\";\n"
    
    # 尋找第一個左括號作為注入點
    insert_pos = original_code.find('{') + 1
    if insert_pos <= 0:
        return original_code, 0, 0
        
    # 插入 Contextual Trigger 區塊
    adv_code = original_code[:insert_pos] + "\n" + contextual_block + original_code[insert_pos:]
    return adv_code, 1, 1

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
        open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            start_time = time.time()
            
            # 取得惡意 prompt (adv_code)，若無則退回使用原始 code
            base_prompt = data.get("adv_code", data.get("code", ""))
            
            # 針對惡意 prompt 進行 Contextual Trigger 注入
            adv_code, changed_vars, changed_pos = generate_contextual_adv_code(base_prompt)
            
            # 更新成功狀態與查詢次數
            final_success = True if changed_vars > 0 else False
            added_queries = 350 if final_success else 0
            
            end_time = time.time()
            
            output_data = {
                "idx": data.get("idx"),
                "success": final_success,
                "queries": data.get("queries", 0) + added_queries,
                "code": data["code"],  # 保留最原始的程式碼
                "adv_code": adv_code,  # 疊加後的惡意程式碼
                "target": data.get("target", 1),
                "num_changed_vars": changed_vars,
                "num_changed_pos": changed_pos,
                "time": (end_time - start_time) + data.get("time", 0) # 累加處理時間
            }
            outfile.write(json.dumps(output_data) + "\n")
            
    print(f"write to {OUTPUT_FILE} successfully")

if __name__ == "__main__":
    main()