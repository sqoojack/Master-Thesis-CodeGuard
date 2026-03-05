import torch

from learning_programs.attacks.summarization.ours import attack, parse_args

if __name__ == "__main__":
    with torch.no_grad():
        attack(parse_args(), train_attack=True)
