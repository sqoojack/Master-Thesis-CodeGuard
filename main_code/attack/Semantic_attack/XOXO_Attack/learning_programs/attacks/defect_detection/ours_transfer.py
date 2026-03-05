import os

import torch

from learning_programs.attacks.defect_detection.ours import par_gcb_transfer_attack, transfer_attack, transfer_parse_args


if __name__ == "__main__":
    with torch.no_grad():
        args = transfer_parse_args()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        attack_fn = par_gcb_transfer_attack if args.surrogate_model == "microsoft/graphcodebert-base" else transfer_attack
        attack_fn(args)
