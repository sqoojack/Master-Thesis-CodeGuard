import json
import os

def load_robust_json(path):
    """
    強大讀取函式：支援標準 JSON 陣列與每一行一個物件的 JSONL 格式。
    """
    if not os.path.exists(path):
        print(f"[!] 找不到檔案: {path}")
        return []
    
    # 嘗試 1: 讀取整個檔案為 JSON (適用於標準 JSON 陣列)
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    # 嘗試 2: 逐行讀取 (適用於標準 JSONL)
    items = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line in ("[", "]", "],"): continue
            try:
                # 處理可能帶有結尾逗號的行 (有些格式錯誤的 JSONL 會這樣)
                if line.endswith(","): line = line[:-1]
                items.append(json.loads(line))
            except json.JSONDecodeError:
                # 略過無法解析的行
                continue
    return items

def merge_datasets():
    # 原始檔案
    xoxo_path = "Dataset/XOXO_attack/XOXO_defect_detection_codebert.jsonl"
    shadowcode_path = "Dataset/ShadowCode/shadowcode_dataset.jsonl"
    
    # 輸出路徑
    output_dir = "Dataset/merged_all"
    output_path = os.path.join(output_dir, "merged_dataset.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    all_data = []

    # 處理 XOXO
    xoxo_items = load_robust_json(xoxo_path)
    x_count = 0
    for entry in xoxo_items:
        # XOXO 格式過濾：必須 success 且有 adv_code
        if entry.get("success") is False or entry.get("adv_code") is None:
            continue
        all_data.append({
            "dataset_source": "XOXO",
            "code": entry.get("code", ""),
            "adv_code": entry.get("adv_code", "")
        })
        x_count += 1
    print(f"[-] XOXO: 成功提取 {x_count} 筆。")

    # 處理 ShadowCode
    shadow_items = load_robust_json(shadowcode_path)
    s_count = 0
    for entry in shadow_items:
        if entry.get("adv_code") is None:
            continue
        all_data.append({
            "dataset_source": "ShadowCode",
            "code": entry.get("code", ""),
            "adv_code": entry.get("adv_code", "")
        })
        s_count += 1
    print(f"[-] ShadowCode: 成功提取 {s_count} 筆。")

    # 寫入合併後的 JSONL (每行一個 JSON)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\n[*] 合併完成！總計 {len(all_data)} 筆。")
    print(f"[*] 輸出檔案: {output_path}")

if __name__ == "__main__":
    merge_datasets()