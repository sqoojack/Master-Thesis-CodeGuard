from learning_programs.attacks.code_generation.ours import attack, parse_args

if __name__ == "__main__":
    attack(parse_args(), train_attack=False, use_logprobs=True)
