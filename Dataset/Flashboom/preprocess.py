"""
python Dataset/Flashboom/preprocess.py \
    --orig_dir main_code/attack/Semantic_attack/Flashboom/data/smartbugs-collection/code \
    --adv_dir main_code/attack/Semantic_attack/Flashboom/data/smartbugs-collection/add_attention_code/MixtralExpert/top0-100 \
    --output Dataset/Flashboom/flashboom_dataset.jsonl \
    --source "Flashboom"
"""
import os
import json
import argparse

def build_flashboom_dataset(orig_dir, adv_dir, output_file, dataset_source):
    dataset = []
    processed_count = 0
    missing_count = 0

    # 遍歷「對抗樣本」目錄 (包含 100 個子資料夾)
    for root, _, files in os.walk(adv_dir):
        for file in files:
            if file.endswith('.sol'):
                adv_path = os.path.join(root, file)
                # 原始檔案都放在 orig_dir 的根目錄下
                orig_path = os.path.join(orig_dir, file)

                if os.path.exists(orig_path):
                    try:
                        with open(orig_path, 'r', encoding='utf-8') as f_orig:
                            orig_code = f_orig.read()
                        
                        with open(adv_path, 'r', encoding='utf-8') as f_adv:
                            adv_code = f_adv.read()

                        # 建立符合目標格式的字典
                        data_entry = {
                            "dataset_source": dataset_source,
                            "code": orig_code,
                            "adv_code": adv_code
                        }
                        
                        dataset.append(data_entry)
                        processed_count += 1
                    except Exception as e:
                        print(f"讀取錯誤 {file}: {e}")
                else:
                    missing_count += 1
                    print(f"找不到原始檔案: {orig_path}")

    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 寫入 JSON Lines (.jsonl) 格式
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for entry in dataset:
            # json.dumps 會自動將換行符號轉為 \n 儲存
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print("-" * 30)
    print("處理完成")
    print(f"成功配對並轉換: {processed_count} 個檔案")
    print(f"缺失原始檔案: {missing_count} 個")
    print(f"結果已儲存至: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Flashboom sol files to JSONL dataset.")
    parser.add_argument("--orig_dir", required=True, help="存放原始漏洞合約的目錄")
    parser.add_argument("--adv_dir", required=True, help="存放攻擊後合約的目錄 (top0-100)")
    parser.add_argument("--output", required=True, help="輸出的 JSONL 檔案路徑")
    parser.add_argument("--source", default="Flashboom", help="dataset_source 的名稱")
    
    args = parser.parse_args()
    
    build_flashboom_dataset(
        orig_dir=args.orig_dir,
        adv_dir=args.adv_dir,
        output_file=args.output,
        dataset_source=args.source
    )