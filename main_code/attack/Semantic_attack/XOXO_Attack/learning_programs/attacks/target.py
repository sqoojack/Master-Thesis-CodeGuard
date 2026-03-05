import argparse
import json
import os
import subprocess

import torch

from learning_programs.attacks.common import Config, get_output_dir, FINETUNING_SCRIPT_PATHS, SUPPORTED_MODELS

FINETUNING_CONFIGS_PATH = "finetuning_configs"

def finetune(dataset: str, model_name: str, seed: int):
    config_path = os.path.join(FINETUNING_CONFIGS_PATH, dataset, f"{model_name}.json")
    with open(config_path) as f:
        config = json.load(f)

    pwd = os.getcwd()
    os.chdir(FINETUNING_SCRIPT_PATHS[dataset])
    cmd = config + ["--seed", str(seed)] + ["--output_dir", get_output_dir(model_name, seed)]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    os.chdir(pwd)


def test(args: argparse.Namespace):
    with torch.no_grad():
        Config(args.dataset).initialize(args)
        args.device = "cuda"
        if args.dataset == "defect_detection":
            from learning_programs.attacks.defect_detection.ours import test_model, BATCH_SIZES
        elif args.dataset == "clone_detection":
            from learning_programs.attacks.clone_detection.ours import test_model, BATCH_SIZES
        elif args.dataset == "summarization":
            from learning_programs.attacks.summarization.ours import test_model, BATCH_SIZES
        args.batch_size = BATCH_SIZES[args.model_name]
        test_model(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dataset", type=str, choices=SUPPORTED_MODELS.keys(), required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.model_name not in SUPPORTED_MODELS[args.dataset]:
        raise ValueError(f"Model {args.model_name} not supported for dataset {args.dataset}")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.finetune:
        finetune(args.dataset, args.model_name, args.seed)
    if args.test:
        test(args)
