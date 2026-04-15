# python Dataset/ShadowCode/preprocess.py
import pandas as pd
import json
import os

def preprocess():
    input_path = "Dataset/ShadowCode/shadowcode_dataset.csv"
    output_path = "Dataset/ShadowCode/shadowcode_dataset_c.jsonl"
    data_source_name = "ShadowCode"
    
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"[!] Error reading CSV: {e}")
        return

    # Filter records for c and python
    target_languages = ['c']
    target_df = df[df['language'].isin(target_languages)]
    
    print(f"[-] Total records in CSV: {len(df)}")
    print(f"[-] Selected records ({', '.join(target_languages)}): {len(target_df)}")

    # Use 'a' mode to append data
    with open(output_path, 'a', encoding='utf-8') as f:
        for _, row in target_df.iterrows():
            full_prompt = str(row['full_prompt']) if not pd.isna(row['full_prompt']) else ""
            perturbation = str(row['perturbation']) if not pd.isna(row['perturbation']) else ""
            malicious_objective = str(row['malicious_objective']) if not pd.isna(row['malicious_objective']) else ""
            
            # Construct clean prompt
            clean_prompt = full_prompt
            if perturbation and perturbation in full_prompt:
                clean_prompt = full_prompt.replace(perturbation, "")
            
            # Define dataset_source as the first key to ensure order
            entry = {
                "dataset_source": data_source_name,
                "code": clean_prompt,
                "adv_code": full_prompt,
                "target": 1,
                "meta": {
                    "case_id": row.get('case_id', 'unknown'),
                    "language": row.get('language', 'unknown'),
                    "perturbation": perturbation,
                    "malicious_objective": malicious_objective
                }
            }
            f.write(json.dumps(entry) + "\n")
            
    print(f"[-] Successfully appended {len(target_df)} records to: {output_path}")

if __name__ == "__main__":
    preprocess()