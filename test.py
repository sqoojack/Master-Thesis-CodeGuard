import json
import os

def prepend_data_source(input_path, output_path):
    """
    Reads a jsonl file, prepends 'dataset_source' to each record, 
    and appends the result to the target jsonl file.
    """
    try:
        # Check if source file exists
        if not os.path.exists(input_path):
            print(f"Error: Source file {input_path} not found.")
            return

        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'a', encoding='utf-8') as outfile:
            
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Load original data
                    data = json.loads(line)
                    
                    # Create a new dictionary with 'dataset_source' at the front
                    # Python 3.7+ dictionaries maintain insertion order
                    updated_data = {"dataset_source": "ShadowCode"}
                    updated_data.update(data)
                    
                    # Append to the target file
                    outfile.write(json.dumps(updated_data, ensure_ascii=False) + '\n')
                except json.JSONDecodeError as e:
                    print(f"Skip invalid JSON line: {e}")
                    
        print(f"Process completed. Data appended to {output_path}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    src = "Dataset/ShadowCode/shadowcode_dataset備份.jsonl"
    dest = "Dataset/ShadowCode/shadowcode_dataset_c.jsonl"
    prepend_data_source(src, dest)