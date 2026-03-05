from learning_programs.attacks.code_generation.ours_limited import parse_args, attack

if __name__ == "__main__":
    attack(parse_args(), train_attack=False, use_logprobs=False)
