"""
python Dataset/ITGen/preprocess.py \
    --input Dataset/ITGen/attack_itgen_all.jsonl \
    --output Dataset/ITGen/itgen_dataset.jsonl \
    --source "ITGen"
"""
import os
import json
import argparse

def process_attack_dataset(input_file, output_file, dataset_source):
    processed_count = 0
    skipped_count = 0

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            orig_code = data.get("Original Code")
            adv_code = data.get("Adversarial Code")
            
            if orig_code is None or adv_code is None:
                skipped_count += 1
                continue
                
            entry = {
                "dataset_source": dataset_source,
                "code": orig_code,
                "adv_code": adv_code
            }
            
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            processed_count += 1
            
    print("-" * 30)
    print("Process completed.")
    print(f"Successfully converted: {processed_count}")
    print(f"Skipped (null values): {skipped_count}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert attack dataset to target JSONL format.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output JSONL file")
    parser.add_argument("--source", default="ITGen", help="Dataset source name")
    
    args = parser.parse_args()
    
    process_attack_dataset(
        input_file=args.input,
        output_file=args.output,
        dataset_source=args.source
    )