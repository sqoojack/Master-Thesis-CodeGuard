# python Dataset/CoTDeceptor/preprocess.py
import os
import json
import argparse

def build_cotdeceptor_dataset(input_file: str, output_file: str, dataset_source: str):
    dataset = []
    processed_count = 0
    error_count = 0

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    # Read and parse the original JSONL file
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line_idx, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Extract required fields
                orig_code = data.get("orig_code", "")
                adv_code = data.get("adv_code", "")

                if not orig_code or not adv_code:
                    print(f"Warning: Missing code fields at line {line_idx + 1}")
                    error_count += 1
                    continue

                # Construct new data entry
                new_entry = {
                    "dataset_source": dataset_source,
                    "code": orig_code,
                    "adv_code": adv_code
                }
                dataset.append(new_entry)
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON at line {line_idx + 1}: {e}")
                error_count += 1

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Write to new JSONL file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for entry in dataset:
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Print execution summary
    print("-" * 30)
    print("Processing completed.")
    print(f"Successfully processed entries: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CoTDeceptor JSONL to unified dataset format.")
    parser.add_argument("--input", required=False, default="CoTDeceptor_orig_dataset.jsonl", help="Input JSONL file path")
    parser.add_argument("--output", required=False, default="CoTDeceptor_dataset.jsonl", help="Output JSONL file path")
    parser.add_argument("--source", default="CoTDeceptor", help="Name for dataset_source")
    
    args = parser.parse_args()
    
    build_cotdeceptor_dataset(
        input_file=args.input,
        output_file=args.output,
        dataset_source=args.source
    )