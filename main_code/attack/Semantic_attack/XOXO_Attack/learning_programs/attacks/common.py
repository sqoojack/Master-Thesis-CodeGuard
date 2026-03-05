import argparse
import os
import sys

RESULTS_DIR = "results/attacks"
OUTPUT_DIR = "saved_models"
PREPROCESS_DIR = "preprocessed"
PREPROCESS_SEED = 2024

CODE_GENERATION_API_MODELS = [
    "openai/gpt-4.1",
    "claude-3-5-sonnet-v2@20241022"
]

CODE_GENERATION_SUPPORTED_MODELS = [
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "mistralai/Codestral-22B-v0.1",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-ai/deepseek-coder-33b-instruct"
] + CODE_GENERATION_API_MODELS

SUPPORTED_MODELS = {
    "defect_detection": ["microsoft/codebert-base", "microsoft/graphcodebert-base", "Salesforce/codet5p-110m-embedding"],
    "clone_detection": ["microsoft/codebert-base", "microsoft/graphcodebert-base", "Salesforce/codet5p-110m-embedding"],
    "summarization": ["microsoft/codebert-base", "Salesforce/codet5p-220m", "uclanlp/plbart-python-en_XX"],
    "humaneval": CODE_GENERATION_SUPPORTED_MODELS,
    "mbpp": CODE_GENERATION_SUPPORTED_MODELS,
    "cweval": CODE_GENERATION_SUPPORTED_MODELS
}

CODE_GENERATION_DATASETS = {"humaneval", "mbpp", "cweval"}

CRITERIA = {
    "defect_detection": "acc",
    "clone_detection": "f1",
    "summarization": "bleu"
}

FINETUNING_SCRIPT_PATHS = {
    "defect_detection": "learning_programs/datasets/CodeXGLUE/Code-Code/Defect-detection/code",
    "clone_detection": "learning_programs/datasets/CodeXGLUE/Code-Code/Clone-detection-BigCloneBench/code",
    "summarization": "learning_programs/datasets/CodeXGLUE/Code-Text/code-to-text/code"
}

LANGUAGES = {
    "defect_detection": "c",
    "clone_detection": "java",
    "summarization": "python",
    "humaneval": "python",
    "mbpp": "python",
    "cweval": "python"
}

def get_dataset_attack() -> tuple[str, str]:
    dataset, attack = os.path.splitext(sys.argv[0])[0].split(os.path.sep)[-2:]
    return dataset, attack


def get_results_dir(dataset: str, attack: str, model_name: str, seed: int) -> str:
    return os.path.join(RESULTS_DIR, dataset, attack, model_name, str(seed))


def get_output_dir(model_name: str, seed: int) -> str:
    return os.path.join(OUTPUT_DIR, str(seed), model_name)


def get_preprocess_path(dataset: str, attack: str) -> str:
    return os.path.join(PREPROCESS_DIR, dataset, attack)


class Config:
    dataset: str
    criterion: str
    language: str
    finetuning_script_path: str | None

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.language = LANGUAGES[self.dataset]
        if self.dataset not in CODE_GENERATION_DATASETS:
            self.criterion = CRITERIA[self.dataset]
            self.finetuning_script_path = FINETUNING_SCRIPT_PATHS[self.dataset]

    def get_model_weights_dir(self, model_name: str, seed: int) -> str:
        return os.path.join(self.finetuning_script_path, get_output_dir(model_name, seed), f"checkpoint-best-{self.criterion}")

    def get_model_weights_path(self, model_name: str, seed: int) -> str:
        path = self.get_model_weights_dir(model_name, seed)
        uses_hf_trainer = self.dataset == "summarization" and model_name in ["Salesforce/codet5p-220m", "uclanlp/plbart-python-en_XX"]
        return path if uses_hf_trainer else os.path.join(path, "model.bin")

    def set_model_args(self, args: argparse.Namespace):
        args.block_size = 512
        if self.dataset == "summarization":
            args.max_source_length = 320
            args.max_target_length = 128
            if args.model_name in ["Salesforce/codet5p-220m", "uclanlp/plbart-python-en_XX"]:
                args.num_beams = 5
            elif args.model_name == "microsoft/codebert-base":
                args.num_beams = 10
        if args.model_name == "microsoft/graphcodebert-base":
            args.data_flow_length = 128
        args.target_weights_dir = self.get_model_weights_dir(args.model_name, args.seed)
        args.target_weights_path = self.get_model_weights_path(args.model_name, args.seed)

    def set_global_vars(self):
        global LANGUAGE
        LANGUAGE = self.language

    def initialize(self, args: argparse.Namespace):
        self.set_global_vars()
        if self.dataset not in CODE_GENERATION_DATASETS:
            self.set_model_args(args)


class AttackConfig(Config):
    attack: str

    def __init__(self, dataset: str, attack: str):
        super().__init__(dataset)
        self.attack = attack

    @classmethod
    def from_script_args(cls):
        """Get the dataset and attack from the script name"""
        return AttackConfig(*get_dataset_attack())

    def supports_model(self, args: argparse.Namespace) -> bool:
        """Check if the model is supported for the given dataset and attack"""
        if self.attack == "ours_transfer":
            return args.surrogate_model in SUPPORTED_MODELS[self.dataset] and args.target_model in SUPPORTED_MODELS[self.dataset]
        if not args.model_name:
            raise ValueError("--model_name is a required argument for non-transfer attacks")
        return args.model_name in SUPPORTED_MODELS[self.dataset]

    def get_results_dir(self, model_name: str, seed: int) -> str:
        """Get the directory to save the results"""
        return get_results_dir(self.dataset, self.attack, model_name, seed)

    def get_target_surrogate_from_model_name(self, model_name: str) -> tuple[str, str]:
        split = model_name.split("/")
        if len(split) != 4:
            raise ValueError(f"--model_name {model_name} must be in the format <target_model>/<surrogate_model for transfer attacks")
        return "/".join(split[:2]), "/".join(split[2:])

    def get_results_path(self, model_name: str, seed: int) -> str:
        """Get the path to save the results"""
        return os.path.join(self.get_results_dir(model_name, seed), "results.jsonl")

    def get_preprocess_path(self) -> str:
        """Get the path to the preprocessed data"""
        return get_preprocess_path(self.dataset, self.attack)

    def add_args(self, parser: argparse.ArgumentParser):
        if self.dataset == "summarization":
            parser.add_argument("--success_criterion", default="rel_threshold", type=str, choices=["zero", "abs_threshold", "rel_threshold", "decrease"])
            parser.add_argument("--bleu_threshold", default=0.75, type=float)

    def set_attack_args(self, args: argparse.Namespace):
        if self.attack == "ours_transfer":
            args.target_model, args.surrogate_model = self.get_target_surrogate_from_model_name(args.model_name)
            if self.dataset not in CODE_GENERATION_DATASETS:
                args.surrogate_model_weights = self.get_model_weights_path(args.surrogate_model, args.seed)
                args.target_model_weights = self.get_model_weights_path(args.target_model, args.seed)
        if self.dataset in CODE_GENERATION_DATASETS:
            args.dataset = self.dataset
        args.results_dir = self.get_results_dir(args.model_name, args.seed)
        args.results_path = self.get_results_path(args.model_name, args.seed)
        args.preprocess_path = self.get_preprocess_path()
    
    def initialize(self, args: argparse.Namespace):
        super().initialize(args)
        self.set_attack_args(args)
        os.makedirs(args.results_dir, exist_ok=True)
        return args


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--model_name", type=str, required=True, help="Model name or path; for transfer attacks, specify as <target_model>/<surrogate_model>.")
        self.add_argument("--eval_batch_size", default=2, type=int)
        self.add_argument('--seed', type=int, required=True)
        self.add_argument("--device", default="cuda", type=str)

    def parse_args(self, *args, **kwargs) -> argparse.Namespace:
        attack_config = AttackConfig.from_script_args()
        attack_config.add_args(self)
        args = super().parse_args(*args, **kwargs)
        if not attack_config.supports_model(args):
            raise ValueError(f"Model {args.model_name} not supported for dataset {attack_config.dataset}")
        attack_config.initialize(args)
        return args
