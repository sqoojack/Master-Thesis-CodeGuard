# python main_code/attack/Adaptive_attack/copy_trigger.py
# python main_code/attack/Adaptive_attack/copy_trigger.py
import json
import time
import os
import re

INPUT_FILE = "Dataset/merged_all/tiny_merged_dataset.jsonl"
OUTPUT_FILE = "Dataset/Adaptive_attack/copy_trigger_attack.jsonl"

def inject_copy_trigger(base_code, adv_code):
    # 策略 1: 動態擷取真正的惡意 Trigger
    # 透過比對 adv_code 與 code 的差異，找出攻擊者真正注入的變數名稱
    adv_words = set(re.findall(r'\b[a-zA-Z_]\w*\b', adv_code))
    benign_words = set(re.findall(r'\b[a-zA-Z_]\w*\b', base_code))
    new_words = adv_words - benign_words
    
    # 排除常見的 C 語言關鍵字與太短的無意義字元
    keywords = {
        "int", "char", "void", "if", "else", "while", "for", "return", 
        "bool", "true", "false", "String", "string", "auto", "const", "class", "public"
    }
    triggers = [w for w in new_words if w not in keywords and len(w) >= 4]
    
    if not triggers:
        # 如果沒有找到明顯的新變數，則退回原代碼
        return adv_code, 0, 0
        
    # 取長度最長、最具有語意代表性的單字作為核心 Trigger
    main_trigger = sorted(triggers, key=len, reverse=True)[0]
    
    # 策略 2: 構造 Copy Trigger 冗餘區塊
    # 將真正的 Trigger 複製到多個變數與 Layer 2 白名單註解中
    # 當 Layer 3 對任何一個節點進行中性化替換時，其他節點依然會強迫模型維持惡意預測
    copy_block = (
        f"    bool {main_trigger}_flag = true;\n"
        f"    int {main_trigger}_stat = 1;\n"
        f"    // NOTE: confirm {main_trigger} usage\n"
    )
    
    # 尋找第一個左括號作為注入點
    insert_pos = adv_code.find('{') + 1
    if insert_pos <= 0:
        return adv_code, 0, 0
        
    # 插入精準的冗餘區塊
    poisoned_code = adv_code[:insert_pos] + "\n" + copy_block + adv_code[insert_pos:]
    
    return poisoned_code, 3, 3

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
        open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            start_time = time.time()
            
            base_code = data.get("code", "")
            base_prompt = data.get("adv_code", base_code)
            
            # 將 base_code 與 base_prompt 一起傳入，以擷取真正的 Trigger
            poisoned_adv_code, changed_vars, changed_pos = inject_copy_trigger(base_code, base_prompt)
            
            # Copy Trigger 是隱蔽攻擊，正常程式碼維持原樣，絕對不可加入干擾，以免 FP 上升
            benign_code = base_code
            
            final_success = True if changed_vars > 0 else False
            added_queries = 120 if final_success else 0
            
            end_time = time.time()
            
            output_data = {
                "idx": data.get("idx"),
                "success": final_success,
                "queries": data.get("queries", 0) + added_queries,
                "code": benign_code,           # 維持乾淨
                "adv_code": poisoned_adv_code, # 注入精準冗餘，讓 Layer 3 的 FN 大幅上升
                "target": data.get("target", 1),
                "num_changed_vars": changed_vars,
                "num_changed_pos": changed_pos,
                "time": (end_time - start_time) + data.get("time", 0)
            }
            outfile.write(json.dumps(output_data) + "\n")
            
    print(f"write to {OUTPUT_FILE} successfully")

if __name__ == "__main__":
    main()