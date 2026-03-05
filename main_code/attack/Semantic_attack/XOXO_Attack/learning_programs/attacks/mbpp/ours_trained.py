from learning_programs.attacks.code_generation.ours import parse_args, attack

if __name__ == "__main__":
    attack(parse_args(), train_attack=True, use_logprobs=True)
