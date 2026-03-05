import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num_seeds", type=int, default=3)
    return parser.parse_args()


def get_seeds(seed: int, num_seeds: int):
    random.seed(seed)
    ret = [random.randint(0, 2 ** 32 - 1) for _ in range(num_seeds)]
    return ret


def main(args: argparse.Namespace):
    for seed in get_seeds(args.seed, args.num_seeds):
        print(seed)


if __name__ == "__main__":
    main(parse_args())
