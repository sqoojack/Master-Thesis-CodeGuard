import argparse
import json

from learning_programs.attacks import common
from learning_programs.attacks.code_generation.ours import parse_args, Model, APIModel
from learning_programs.datasets.code_generation import load_examples
from learning_programs.runners.docker.runner import TestRunnerManager as TRMDocker
from learning_programs.runners.python.runner import TestRunnerManager as TRMPython


def main(args: argparse.Namespace):
    print("Loading target model...")
    with (TRMDocker if args.dataset == "cweval" else TRMPython)(args.max_workers) as tmgr:
        if args.model_name in common.CODE_GENERATION_API_MODELS:
            target = APIModel(args.model_name, tmgr)
        else:
            target = Model(args.model_name, tmgr, args.allow_multi_gpu)
        examples = load_examples(args.dataset)
        _, generations, results, _ = target.generate_test(examples)
        num_passed = 0
        with open(args.results_path, "w") as f:
            for example, generation, result in zip(examples, generations, results):
                if not (result[0].functional and result[0].secure):
                    continue
                f.write(json.dumps({
                    "task_id": example.idx,
                    "generated_code": generation,
                }) + "\n")
                num_passed += 1
        print(f"Filtered {num_passed}/{len(examples)} examples that the target model got right")


if __name__ == "__main__":
    main(parse_args())
