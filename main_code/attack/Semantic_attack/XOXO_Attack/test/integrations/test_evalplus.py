from collections import defaultdict

from evalplus.data import get_human_eval_plus, get_mbpp_plus

from learning_programs.transforms.python.processor import Processor
from learning_programs.transforms.transform import merge_transforms


def main():
    processor = Processor()
    for name, problem in (get_human_eval_plus() | get_mbpp_plus()).items():
        code = problem["prompt"] + problem["canonical_solution"]
        print("-" * 80)
        print(f"Problem: {name}")
        print("-" * 80)
        print(code)

        transforms = processor.find_statement_transforms(code)
        print(f"Transforms: {len(transforms)}")

        transforms_by_count = defaultdict(int)
        for t in transforms:
            transforms_by_count[t.name] += 1

        for i, (name, count) in enumerate(
            sorted(transforms_by_count.items(), key=lambda x: x[1], reverse=True)
        ):
            print(f"{i + 1}.  {name}: {count}")

        print()

        ranges = merge_transforms(transforms)
        print(f"Transform ranges: {len(ranges)}")

        for i, r in enumerate(ranges):
            print(f"{i + 1}.  {r}")

        print()

        transforms = processor.find_function_transforms(code)
        print(f"Function transforms: {len(transforms)}")

        print()

        transforms_by_count = defaultdict(int)
        for t in transforms:
            transforms_by_count[t.name] += 1

        for i, (name, count) in enumerate(
            sorted(transforms_by_count.items(), key=lambda x: x[1], reverse=True)
        ):
            print(f"{i + 1}.  {name}: {count}")

        print()

        identifiers = processor.find_identifiers(code)
        print(f"Identifiers: {len(identifiers)}")
        for i, id in enumerate(identifiers):
            print(f"{i + 1}.  {id.name.decode()}")

        print()


if __name__ == "__main__":
    main()
