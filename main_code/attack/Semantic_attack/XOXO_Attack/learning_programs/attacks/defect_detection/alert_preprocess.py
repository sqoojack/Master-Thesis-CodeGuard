import argparse

from learning_programs.attacks.common import get_preprocess_path, Config
from learning_programs.attacks.defect_detection.alert import preprocess
from learning_programs.datasets.defect_detection import load_examples


DATASET = "defect_detection"


if __name__ == "__main__":
    Config(DATASET).set_global_vars()
    args = argparse.Namespace()
    args.preprocess_path = get_preprocess_path(DATASET, "alert")
    args.from_scratch = True
    args.block_size = 512
    args.device = "cuda"
    preprocess(load_examples("test"), args)
