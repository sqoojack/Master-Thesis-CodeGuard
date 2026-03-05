import json
import os
import shutil
from pathlib import Path

ATTACK = "ours_no_logprobs"
MODEL_NAME  = "Qwen/CodeQwen1.5-7B-Chat"
SEED = "2024"
CODE_UNDER_TEST_SOURCEDIR = "TODO"
TEST_SOURCEDIR = Path("test_sourcedir")
TEST_WORKDIR = Path("test_workdir")

def load_tests():
    test_files = dict()
    for dirent in os.scandir(TEST_SOURCEDIR):
        if dirent.name.endswith("_test.py"):
            module_name = dirent.name.replace("_test.py", "_task")
            with open(dirent.path, "r") as f:
                file = f.read()
            test_files[module_name] = file
    return test_files

test_files = load_tests()

if os.path.exists(TEST_WORKDIR):
    shutil.rmtree(TEST_WORKDIR)
os.makedirs(TEST_WORKDIR)

with open(f"{ATTACK}/{MODEL_NAME}/{SEED}/results.jsonl") as f:
    for line in f:
        result = json.loads(line)
        name = result["example"]["task_id"].split("_", 1)[1]
        module_name = name + "_task"
        file_name = f"{module_name}.py"
        for i, code in enumerate(result["result"][3]):
            code = code[0]
            new_module_name = f"{name}_{i}_task"
            new_test_name = f"{name}_{i}_test"
            test_file = test_files[module_name].replace(f"from {module_name} ", f"from {new_module_name} ", 1)
            with open(TEST_WORKDIR / f"{new_module_name}.py", "w") as f:
                f.write(code)
            with open(TEST_WORKDIR / f"{new_test_name}.py", "w") as f:
                f.write(test_file)
            if i == 250:
                break
        break
