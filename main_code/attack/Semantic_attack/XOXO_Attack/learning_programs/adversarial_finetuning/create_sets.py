import argparse
import subprocess
import os


model_names = {
    "microsoft/codebert-base": "codebert",
    "Salesforce/codet5p-110m-embedding": "codet5",
    "microsoft/graphcodebert-base": "graphcodebert"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, choices=["rnns", "ours"], required=True)
    parser.add_argument("--model_name", type=str, choices=["microsoft/codebert-base", "microsoft/graphcodebert-base", "Salesforce/codet5p-110m-embedding"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--split", type=str, required=True, help="Dataset split to attack", choices=["train", "valid"])
    return parser.parse_args()


def main(args):
    for part, gpu in enumerate(args.gpus):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        session_name = f"{args.attack}_{model_names[args.model_name]}_{args.seed}_{args.split}_{part}"
        cmd = [
            "screen",
            "-S",
            session_name,
            "-L",
            "-Logfile",
            f"logs/{session_name}.log",
            "-dm",
            "python",
            "-m",
            f"learning_programs.attacks.defect_detection.{args.attack}_adv",
            "--model",
            args.model_name,
            "--seed",
            str(args.seed),
            "--part",
            str(part),
            "--split",
            str(args.split)
        ]
        print(f"Running command: {' '.join(cmd)} on GPU {gpu}")
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main(parse_args())