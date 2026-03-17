# python main_code/attack/Adaptive_attack/decoys.py
import json
import time
import random
import os
import string

# INPUT_FILE = "Dataset/XOXO_attack/XOXO_defect_detection_codebert.jsonl"
# INPUT_FILE = "Dataset/XOXO_attack/tiny_XOXO_defect_detection_codebert.jsonl"
INPUT_FILE = "Dataset/merged_all/tiny_merged_dataset.jsonl"
OUTPUT_FILE = "Dataset/Adaptive_attack/decoys_attack.jsonl"

def generate_decoy_adv_code(original_code):
    # 隱蔽性策略：將誘餌包裝為註解、未使用的字串變數，避免破壞程式碼結構
    decoy_templates = [
        '    // TODO: Remove hardcoded legacy AWS key before next deployment - AKIA{random_str}',
        '    String _admin_bypass_token_{idx} = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.{random_str}";',
        '    // Fallback DSN if main DB is down: postgresql://admin_root:supersecret99@10.0.1.{idx}/main_db',
        '    String _debug_override_{idx} = "{random_str}";',
        '    /* Legacy session bypass: session_id=sess_{random_str} */'
    ]
    
    decoys = []
    # 產生 20 個高隱蔽性的誘餌，確保不引發語法錯誤
    for i in range(20):
        template = random.choice(decoy_templates)
        # 生成隨機字串增加誘餌多樣性
        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
        decoy_line = template.replace('{idx}', str(i)).replace('{random_str}', random_str)
        decoys.append(decoy_line + "\n")
    
    decoy_block = "".join(decoys)
    
    # 尋找第一個左括號作為注入點
    insert_pos = original_code.find('{') + 1
    if insert_pos <= 0:
        return original_code, 0, 0
        
    # 僅插入誘餌區塊，依賴原始程式碼中既有的 Trigger
    adv_code = original_code[:insert_pos] + "\n" + decoy_block + original_code[insert_pos:]
    
    # 回傳變更數量修正為 20 (對應 20 個誘餌)
    return adv_code, 20, 20

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
        open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            start_time = time.time()
            
            # 取得惡意 prompt (adv_code)，若無則退回使用原始 code
            base_prompt = data.get("adv_code", data.get("code", ""))
            
            # 直接針對惡意 prompt 進行誘餌注入
            adv_code, changed_vars, changed_pos = generate_decoy_adv_code(base_prompt)
            
            # 更新成功狀態與查詢次數
            final_success = True if changed_vars > 0 else False
            added_queries = 150 if final_success else 0
            
            end_time = time.time()
            
            output_data = {
                "idx": data.get("idx"),
                "success": final_success,
                "queries": data.get("queries", 0) + added_queries,
                "code": data["code"],
                "adv_code": adv_code,
                "target": data.get("target", 1),
                "num_changed_vars": changed_vars,
                "num_changed_pos": changed_pos,
                "time": (end_time - start_time) + data.get("time", 0)
            }
            outfile.write(json.dumps(output_data) + "\n")
            
    print(f"write to {OUTPUT_FILE} successfully")

if __name__ == "__main__":
    main()