import torch

from learning_programs.attacks.defect_detection.ours import par_gcb_attack, attack, parse_args

if __name__ == "__main__":
    with torch.no_grad():
        args = parse_args()
        attack_fn = par_gcb_attack if args.model_name == "microsoft/graphcodebert-base" else attack
        attack_fn(args, train_attack=True)
