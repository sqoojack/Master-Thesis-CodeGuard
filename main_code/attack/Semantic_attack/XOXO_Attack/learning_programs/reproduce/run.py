from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import subprocess
import threading
import torch
from collections import deque
from datetime import datetime
from typing import NamedTuple

import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from learning_programs.attacks.common import AttackConfig, SUPPORTED_MODELS, LANGUAGES, RESULTS_DIR, CODE_GENERATION_DATASETS, CODE_GENERATION_API_MODELS
from learning_programs.metrics.codebleu import codebleu
from learning_programs.reproduce.get_seeds import get_seeds

DEFAULT_SEED = 2024
DEFAULT_NUM_SEEDS = 5
DATASETS = sorted(SUPPORTED_MODELS.keys())
MODELS = sorted({model_name for model_names in SUPPORTED_MODELS.values() for model_name in model_names})
ATTACKS = ["alert", "mhm", "rnns", "wir", "ours", "ours_trained", "ours_no_logprobs", "ours_limited", "ours_limited_no_logprobs"]
ALLOW_MULTI_GPU = False
MAX_WORKERS_PER_GPU = multiprocessing.cpu_count() // torch.cuda.device_count() if torch.cuda.is_available() else 1

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for the random number generator that is used to generate the seeds for the experiments.")
    parser.add_argument("--num_seeds", type=int, default=DEFAULT_NUM_SEEDS, help="Number of seeds to generate (one per experiment).")
    parser.add_argument("--seeds", nargs="+", type=int, help="Seeds to use for the experiments (overrides --seed and --num_seeds).")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS, help="Datasets to run the experiments on.")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS, help="Models to run the experiments on. Note that the models must be supported by the attack.")
    parser.add_argument("--attacks", nargs="+", default=ATTACKS, choices=ATTACKS, help="Attacks to run the experiments with.")
    parser.add_argument("--all", action="store_true", help="Run the complete evaluation pipeline.")
    parser.add_argument("--dataset_preprocess", action="store_true", help="Preprocess the specified datasets.")
    parser.add_argument("--finetune", action="store_true", help="Finetune the specified models on the specified datasets.")
    parser.add_argument("--test", action="store_true", help="Test the specified models on the specified datasets.")
    parser.add_argument("--attack_preprocess", action="store_true", help="Preprocess the data for the specified attacks on the specified datasets.")
    parser.add_argument("--attack", action="store_true", help="Run the attack on the specified models on the specified datasets.")
    parser.add_argument("--process_results", action="store_true", help="Process the results for the specified models on the specified datasets.")
    parser.add_argument("--latex", action="store_true", help="Print the results in LaTeX format.")
    return parser.parse_args()


class SupportedCombination(NamedTuple):
    """A supported combination of dataset, model, and seed."""
    dataset: str
    model: str
    seed: int


def supports_multi_gpu(attack: str, dataset: str) -> bool:
    """Return whether the specified attack supports multi-GPU training on the specified dataset."""
    if dataset in CODE_GENERATION_DATASETS:
        return attack in {"ours", "ours_trained"}
    return attack == "ours_trained" and dataset != "summarization"


def visible_gpus_env(gpus: list[int]) -> dict[str, str]:
    """Return an environment with CUDA_VISIBLE_DEVICES set to the specified GPUs."""
    return os.environ | {"CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpus)}


def attack_cmd(attack: str, comb: SupportedCombination) -> list[str]:
    """Return the command to run the specified attack on the specified combination."""
    cmd = ["python", "-m", f"learning_programs.attacks.{comb.dataset}.{attack}", "--model", comb.model, "--seed", str(comb.seed)]
    if ALLOW_MULTI_GPU and supports_multi_gpu(attack, comb.dataset):
        cmd.append("--allow_multi_gpu")
    elif comb.dataset in CODE_GENERATION_DATASETS:
        cmd.append("--max_workers")
        cmd.append(str(MAX_WORKERS_PER_GPU))
    return cmd


def attack_preprocess_cmd(attack: str, dataset: str) -> list[str]:
    """Return the command to preprocess the data for the specified attack on the specified combination."""
    return ["python", "-m", f"learning_programs.attacks.{dataset}.{attack}_preprocess"]


def finetune_cmd(comb: SupportedCombination) -> list[str]:
    """Return the command to finetune the specified combination."""
    return ["python", "-m", "learning_programs.attacks.target", "--finetune", "--dataset", comb.dataset, "--model", comb.model, "--seed", str(comb.seed)]


def test_cmd(comb: SupportedCombination) -> list[str]:
    """Return the command to finetune the specified combination."""
    return ["python", "-m", "learning_programs.attacks.target", "--test", "--dataset", comb.dataset, "--model", comb.model, "--seed", str(comb.seed)]


def dataset_preprocess_cmd(dataset: str) -> list[str]:
    """Return the command to preprocess the specified dataset."""
    return ["python", "-m", "learning_programs.datasets.preprocess", f"--{dataset}"]


def get_supported_combinations(datasets: list[str], models: list[str], seeds: list[int]) -> list[SupportedCombination]:
    """Return a list of supported combinations."""
    combs = []
    for seed in seeds:
        for dataset in datasets:
            for model_name in models:
                if model_name not in SUPPORTED_MODELS[dataset]:
                    logger.info(f"Model {model_name} is not supported on dataset {dataset}. Skipping.")
                    continue
                combs.append(SupportedCombination(dataset, model_name, seed))
    return combs


def get_timestamp() -> str:
    """Return a timestamp string."""
    return datetime.now().strftime("%Y%m%d%H%M%S")


def get_visible_gpus() -> list[int]:
    """Return the visible GPUs."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return list(range(torch.cuda.device_count()))
    return [int(gpu) for gpu in visible_devices.split(",")]


class Machine:
    """A machine that can run tasks."""
    gpu_count: int
    gpu_pool: list[int]
    gpu_sema: threading.Semaphore
    task_queue: deque[Task]
    logger = logging.Logger
    logs_dir: str = "logs"
    machine_log_path: str = "logs/machine.log"

    def __init__(self):
        self.gpu_pool = sorted(get_visible_gpus(), reverse=True)
        self.gpu_count = len(self.gpu_pool)
        self.gpu_sema = threading.Semaphore(self.gpu_count)
        self.task_queue = deque()
        self.logger = logging.getLogger(str(self))
        os.makedirs(self.logs_dir, exist_ok=True)
        self.logger.info(f"Initialized machine with {self.gpu_count} GPUs.")

    def acquire_gpu(self) -> int:
        """Acquire a GPU from the pool."""
        self.gpu_sema.acquire()
        return self.gpu_pool.pop()

    def acquire_gpus(self, n: int) -> list[int]:
        """Acquire all n GPUs from the pool."""
        return [self.acquire_gpu() for _ in range(n)]

    def release_gpus(self, gpus: list[int]):
        """Release the specified GPUs back into the pool."""
        for gpu in gpus:
            self.release_gpu(gpu)
    
    def release_gpu(self, gpu: int):
        """Release the specified GPU back into the pool."""
        self.gpu_pool.append(gpu)
        self.gpu_pool.sort(reverse=True)
        self.gpu_sema.release()

    def add_task(self, task: Task):
        """Add a task to the task queue."""
        self.task_queue.append(task)

    def pop_task(self) -> Task:
        """Pop a task from the task queue."""
        return self.task_queue.popleft()

    def dispatch_tasks(self):
        """Dispatch all tasks in the task queue and wait until they are done."""
        while self.task_queue:
            task = self.pop_task()
            self.logger.info(f"Dispatching {task}")
            task.dispatch()
        self.wait_until_done()

    def wait_until_done(self):
        """Wait until all tasks are done."""
        self.logger.info("Waiting until all tasks are done.")
        gpus = self.acquire_gpus(self.gpu_count)
        self.logger.info(f"Reclaimed all GPUs {gpus} from tasks.")
        self.release_gpus(gpus)
        self.logger.info(f"Released all GPUs {gpus} back into the pool.")

    def __str__(self):
        return f"Machine[gpu_count={self.gpu_count}, logs_dir={self.logs_dir}, machine_log_path={self.machine_log_path}]"


class Task:
    """A task that can be dispatched to a machine."""
    name: str
    machine: Machine
    comb: SupportedCombination
    logs_path: str
    time_start: str
    time_end: str
    logger = logging.Logger
    success: bool | None
    num_gpus_needed: int | None = None
    cmd: str = ""

    def __init__(self, machine: Machine, comb: SupportedCombination):
        self.machine = machine
        self.comb = comb
        self.logger = logging.getLogger(str(self))
        self.setup_logs_path()
        self.success = None

    def setup_logs_path(self):
        """Set up the logs path for this task."""
        self.logs_path = os.path.join(self.machine.logs_dir, self.name, self.comb.dataset, self.comb.model, f"{self.comb.seed}.log")
        os.makedirs(os.path.dirname(self.logs_path), exist_ok=True)
    
    def _dispatch(self, gpus: list[int]):
        """Run the specified command and log the output."""
        self.logger.info(f"Running command {self.cmd}")
        with open(self.logs_path, "w") as f:
            self.logger.info(f"Logging to {self.logs_path}")
            self.time_start = datetime.now().strftime("%Y%m%d%H%M%S")
            self.success = subprocess.run(self.cmd, stdout=f, stderr=f, env=visible_gpus_env(gpus)).returncode == 0
            self.time_end = datetime.now().strftime("%Y%m%d%H%M%S")
        self.logger.info(f"Task finished, status: {self.status()}")
        with open(self.machine.machine_log_path, "a") as f:
            f.write(json.dumps(self._asdict()) + "\n")
        self.machine.release_gpus(gpus)
        self.logger.info(f"Released GPUs {gpus}")

    def status(self) -> str:
        """Return the status of the task."""
        if self.success is None:
            return "Pending"
        return "Success" if self.success else "Failed"

    def dispatch(self):
        """Dispatch the task to the machine in a new thread once the required GPUs are acquired."""
        gpus = self.machine.acquire_gpus(self.num_gpus_needed)
        self.logger.info(f"Acquired GPUs {gpus}")
        threading.Thread(target=self._dispatch, args=(gpus,)).start()

    def _asdict(self):
        """Return a dictionary representation of this task."""
        return {
            "time_start": self.time_start,
            "time_end": self.time_end,
            "name": self.name,
            "comb": self.comb._asdict(),
            "logs_path": self.logs_path,
            "cmd": self.cmd,
            "success": self.success
        }
    
    def __str__(self):
        return f"Task[{self.name}, {self.comb}]"


class FinetuneTask(Task):
    """A finetuning task."""
    name: str = "finetune"

    def __init__(self, machine: Machine, comb: SupportedCombination):
        self.cmd = finetune_cmd(comb)
        self.num_gpus_needed = machine.gpu_count
        super().__init__(machine, comb)


class TestTask(Task):
    """A test task."""
    name: str = "test"

    def __init__(self, machine: Machine, comb: SupportedCombination):
        self.cmd = test_cmd(comb)
        self.num_gpus_needed = 1 if comb.dataset == "summarization" else machine.gpu_count
        super().__init__(machine, comb)


class AttackTask(Task):
    """An attack task."""
    def __init__(self, machine: Machine, comb: SupportedCombination, attack: str):
        self.name = attack
        self.cmd = attack_cmd(attack, comb)
        self.num_gpus_needed = machine.gpu_count if ALLOW_MULTI_GPU and supports_multi_gpu(attack, comb.dataset) else 1
        super().__init__(machine, comb)


class AttackPreprocessTask(Task):
    """An attack preprocess task."""
    attack: str
    dataset: str
    name: str = "attack_preprocess"

    def __init__(self, machine: Machine, dataset: str, attack: str):
        self.attack = attack
        self.dataset = dataset
        self.cmd = attack_preprocess_cmd(attack, dataset)
        self.num_gpus_needed = 1
        self.machine = machine
        self.logger = logging.getLogger(str(self))
        self.setup_logs_path()
        self.success = None

    def setup_logs_path(self):
        """Set up the logs path for this task."""
        self.logs_path = os.path.join(self.machine.logs_dir, self.name, self.attack, f"{self.dataset}.log")
        os.makedirs(os.path.dirname(self.logs_path), exist_ok=True)

    def _asdict(self):
        """Return a dictionary representation of this task."""
        return {
            "time_start": self.time_start,
            "time_end": self.time_end,
            "name": self.name,
            "attack": self.attack,
            "dataset": self.dataset,
            "logs_path": self.logs_path,
            "cmd": self.cmd,
            "success": self.success
        }

    def __str__(self):
        return f"Task[{self.name}, {self.attack}, {self.dataset}]"


class DatasetPreprocessTask(Task):
    """A dataset preprocess task."""
    dataset: str
    name: str = "dataset_preprocess"

    def __init__(self, machine: Machine, dataset: str):
        self.dataset = dataset
        self.cmd = dataset_preprocess_cmd(dataset)
        self.num_gpus_needed = 1 # Dataset preprocessing does not require GPUs, but we need to acquire one to block other tasks
        self.machine = machine
        self.logger = logging.getLogger(str(self))
        self.setup_logs_path()
        self.success = None

    def setup_logs_path(self):
        """Set up the logs path for this task."""
        self.logs_path = os.path.join(self.machine.logs_dir, self.name, f"{self.dataset}.log")
        os.makedirs(os.path.dirname(self.logs_path), exist_ok=True)

    def _asdict(self):
        """Return a dictionary representation of this task."""
        return {
            "time_start": self.time_start,
            "time_end": self.time_end,
            "name": self.name,
            "dataset": self.dataset,
            "logs_path": self.logs_path,
            "cmd": self.cmd,
            "success": self.success
        }

    def __str__(self):
        return f"Task[{self.name}, {self.dataset}]"



def fetch_models(models: list[str]):
    """Fetch the specified models."""
    for model in models:
        if model in CODE_GENERATION_API_MODELS:
            continue
        logger.info(f"Fetching model {model}")
        subprocess.run(["huggingface-cli", "download", model], check=True)



def collect_results(combs: list[SupportedCombination], attacks: list[str]) -> pd.DataFrame:
    """Collect the results for the specified combinations and attacks."""
    all_results = []
    collected = 0
    failed_to_decode = 0
    total = len(combs) * len(attacks)
    with tqdm(total=total) as pbar:
        for comb in combs:
            for attack in attacks:
                results_path = AttackConfig(comb.dataset, attack).get_results_path(comb.model, comb.seed)
                if not os.path.exists(results_path):
                    pbar.write(f"No results found for {comb} with attack {attack}.")
                    pbar.update(1)
                    continue
                with open(results_path) as f:
                    try:
                        results = [json.loads(line) | comb._asdict() | {"attack": attack} for line in f]
                        all_results.extend(results)
                        collected += 1
                    except json.JSONDecodeError:
                        pbar.write(f"Failed to decode results for {comb} with attack {attack}.")
                        failed_to_decode += 1
                pbar.update(1)
    logger.info(f"Collected {collected}/{total} results.")
    logger.info(f"Failed to decode {failed_to_decode}/{total} results.")
    return pd.DataFrame.from_records(all_results)


MODEL_NAME_PAPER_MAP = {
    "microsoft/codebert-base": "\codebert",
    "microsoft/graphcodebert-base": "\graphcodebert",
    "Salesforce/codet5p-110m-embedding": "\codetpshort",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "\qwenlargeshort",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "\qwensmallshort",
    "meta-llama/Llama-3.1-8B-Instruct": "\llamashort",
    "deepseek-ai/deepseek-coder-6.7b-instruct": "\deepseekshort",
    "deepseek-ai/deepseek-coder-33b-instruct": "\deepseeklargeshort",
    "mistralai/Codestral-22B-v0.1": "\codestralshort",
    "claude-3-5-sonnet-v2@20241022": "\claude",
}

MODEL_NAME_SHORT_MAP = {
    "microsoft/codebert-base": "CodeBERT",
    "microsoft/graphcodebert-base": "GraphCodeBERT",
    "Salesforce/codet5p-110m-embedding": "CodeT5+",
    
    "Qwen/Qwen2.5-Coder-32B-Instruct": "Qwen2.5-Coder-32B",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen2.5-Coder-7B",
    "mistralai/Codestral-22B-v0.1": "Codestral-22B",
    "deepseek-ai/deepseek-coder-6.7b-instruct": "DeepSeek-6.7B",
    "deepseek-ai/deepseek-coder-33b-instruct": "DeepSeek-33B",
    "meta-llama/Llama-3.1-8B-Instruct": "Meta-Llama-3.1-8B",
    "claude-3-5-sonnet-v2@20241022": "Claude-3-5",
}


ATTACK_NAME_PAPER_MAP = {
    "alert": "\\alert",
    "mhm": "\mhm",
    "rnns": "\\rnns",
    "wir": "\wir",
    "ours": "\\alg",
    "ours_trained": "\\algw",
    "ours_no_logprobs": "\\alg noP",
    "ours_limited": "\\algp",
    "ours_limited_no_logprobs": "\\alg noP",
}


ATTACK_NAME_SHORT_MAP = {
    "alert": "ALERT",
    "mhm": "MHM",
    "rnns": "RNNS",
    "wir": "WIR",
    "ours": "GCGS+P",
    "ours_trained": "GCGS+W",
    "ours_no_logprobs": "GCGS",
    "ours_limited": "OURS-L",
    "ours_limited_no_logprobs": "OURS-LNP",
}


def process_results(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    COLS = ["success", "queries", "time", "num_changed_vars", "num_changed_pos"]
    STATS_ALL_RUNS = "mean"
    STATS_ACROSS_SEEDS = ["mean", "std"]
    # Copy the DataFrame to avoid modifying the original
    df = df[df.dataset == dataset].copy()
    # if dataset == "clone_detection":
        # # Mark more than 2000 queries as unsuccessful
        # df.loc[(df.queries > 2000) & (df.attack.isin({"ours", "ours_trained"})), "success"] = False
        # df.loc[(df.queries > 2000) & (df.attack.isin({"ours", "ours_trained"})), "queries"] = 2000
    # Computing the CodeBLEU score
    print("Computing CodeBLEU scores")
    code_map = dict()
    with tqdm(total=len(df.model.unique()) * len(df.attack.unique()) * len(df.seed.unique())) as pbar:
        for m in df.model.unique():
            for a in df.attack.unique():
                for s in df.seed.unique():
                    cdf = df[(~df.adv_code.isna()) & (df.model == m) & (df.attack == a) & (df.seed == s)][["code", "adv_code"]]
                    code_map[(m, a, s)] = codebleu.compute_codebleu(cdf.code.to_list(), cdf.adv_code.to_list(), LANGUAGES[dataset])
                    pbar.update(1)
    # Mark parser errors as unsuccessful
    df.loc[df.success.isna(), "success"] = False
    # Group by model, attack, and seed, and aggregate statistics across all runs
    df = df.groupby(["model", "attack", "seed"], as_index=False).agg({col: STATS_ALL_RUNS for col in COLS})
    codebleu_df = pd.DataFrame([(m, a, s, c) for (m, a, s), c in code_map.items()], columns=["model", "attack", "seed", "codebleu"])
    df = df.merge(codebleu_df, on=["model", "attack", "seed"])
    COLS += ["codebleu"]
    # Group by model and attack, and aggregate statistics across seeds
    df = df.groupby(["model", "attack"], as_index=False).agg({col: STATS_ACROSS_SEEDS for col in COLS})
    # Change success to percentage
    df["success"] = df["success"] * 100
    # Change CodeBLEU to percentage
    df["codebleu"] = df["codebleu"] * 100
    # Round the statistics to two decimal places
    # df = df.astype({(col, stat): float for col in COLS for stat in STATS_ACROSS_SEEDS}).round(2)
    df = df.astype({(col, stat): float for col in COLS for stat in STATS_ACROSS_SEEDS})
    return df


def fmt_mean_std(df: pd.DataFrame, col: str, num_dec: int, latex: bool) -> pd.Series:
    def _round(x: float) -> str:
        return f"{x:.{num_dec}f}"
    mean_series = df[col, "mean"].apply(_round)
    std_series = df[col, "std"].apply(_round)
    if not latex:
        return mean_series + " ±" + std_series
    for model in df.model.unique():
        # Highlight the best value in each column
        if col in {"success", "codebleu"}:
            best_mean = df[df.model == model][col, "mean"].idxmax()
        elif col in {"queries", "time", "num_changed_vars", "num_changed_pos"}:
            best_mean = df[df.model == model][col, "mean"].idxmin()
        else:
            raise ValueError(f"Unknown column {col}")
        mean_series[best_mean] = "\\textbf{" + mean_series[best_mean] + "}"
    return mean_series + " \\tiny ±" + std_series


def fmt_asr_qr_table(df: pd.DataFrame, dataset: str, latex: bool) -> pd.DataFrame:
    model = df.model.apply(lambda x: (MODEL_NAME_PAPER_MAP if latex else MODEL_NAME_SHORT_MAP)[x])
    attack = df.attack.apply(lambda x: (ATTACK_NAME_PAPER_MAP if latex else ATTACK_NAME_SHORT_MAP)[x])
    asr = fmt_mean_std(df, "success", 2, latex)
    qr = fmt_mean_std(df, "queries", 0, latex)
    new_df = pd.DataFrame({"Model": model, "Attack": attack, "ASR": asr, "QR": qr})
    if dataset not in CODE_GENERATION_DATASETS:
        new_df = new_df.pivot(index="Attack", columns="Model", values=["ASR", "QR"])
        # new_df = new_df[[(metric, model) for model in ["CodeBERT", "GraphCodeBERT", "CodeT5+"] for metric in ["ASR", "QR"]]]
    return tabulate(new_df, headers='keys', tablefmt="latex_raw" if latex else "simple", floatfmt=".1f", stralign="right")


def fmt_nv_np_cb_table(df: pd.DataFrame, dataset: str, latex: bool) -> pd.DataFrame:
    model = df.model.apply(lambda x: (MODEL_NAME_PAPER_MAP if latex else MODEL_NAME_SHORT_MAP)[x])
    attack = df.attack.apply(lambda x: (ATTACK_NAME_PAPER_MAP if latex else ATTACK_NAME_SHORT_MAP)[x])
    nv = fmt_mean_std(df, "num_changed_vars", 2, latex)
    np = fmt_mean_std(df, "num_changed_pos", 2, latex)
    cb = fmt_mean_std(df, "codebleu", 2, latex)
    new_df = pd.DataFrame({"Model": model, "Attack": attack, "NV": nv, "NP": np, "CB": cb})
    if dataset not in CODE_GENERATION_DATASETS:
        new_df = new_df.pivot(index="Attack", columns="Model", values=["NV", "NP", "CB"])
        # new_df = new_df[[(metric, model) for model in ["CodeBERT", "GraphCodeBERT", "CodeT5+"] for metric in ["NV", "NP", "CB"]]]
    return tabulate(new_df, headers='keys', tablefmt="latex_raw" if latex else "simple", floatfmt=".1f", stralign="right")


def main(args: argparse.Namespace):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)

    if args.seeds:
        logger.info("Detected --seeds argument, ignoring --seed and --num_seeds.")
        seeds = args.seeds
        logger.info(f"Using the following seeds: {args.seeds}")
    else:
        logger.info(f"Generating {args.num_seeds} seeds from seed {args.seed}.")
        seeds = get_seeds(args.seed, args.num_seeds) if args.seeds is None else args.seeds
        logger.info(f"Generated the following seeds: {seeds}")

    if args.all or args.dataset_preprocess or args.finetune or args.test or args.attack_preprocess or args.attack:
        machine = Machine()

    if args.all or args.dataset_preprocess:
        preprocessed_summarization = False
        for dataset in args.datasets:
            machine.add_task(DatasetPreprocessTask(machine, dataset))
            # If we are preprocessing a code generation dataset, we need to preprocess summarization first (used to mine identifiers for our attack)
            if dataset == "summarization":
                preprocessed_summarization = True
            elif dataset in CODE_GENERATION_DATASETS and not preprocessed_summarization:
                machine.add_task(DatasetPreprocessTask(machine, "summarization")) #
        machine.dispatch_tasks() # All specified datasets need to be preprocessed before finetuning, testing, and attacking

    combs = get_supported_combinations(args.datasets, args.models, seeds)

    if args.all or args.finetune or args.test or args.attack_preprocess or args.attack:
        fetch_models(list({comb.model for comb in combs})) # Fetch all models before running any tasks

    if args.all or args.finetune:
        for comb in combs:
            if comb.dataset not in CODE_GENERATION_DATASETS: # We do not need to finetune models for code generation tasks
                machine.add_task(FinetuneTask(machine, comb))
        machine.dispatch_tasks() # All specified models need to be finetuned before testing or attacking

    if args.all or args.test:
        for comb in combs:
            if comb.dataset not in CODE_GENERATION_DATASETS: # We do not need to test models for code generation tasks
                machine.add_task(TestTask(machine, comb))
        machine.dispatch_tasks() # All specified models need to be tested before attacking

    if args.all or args.attack_preprocess:
        for dataset in args.datasets:
            if dataset not in CODE_GENERATION_DATASETS: # We do not need to preprocess data for code generation tasks
                for attack in args.attacks:
                    if attack == "alert": # Only alert attacks need special preprocessing
                        machine.add_task(AttackPreprocessTask(machine, dataset, attack))
        machine.dispatch_tasks() # All specified attacks need to be preprocessed before attacking

    if args.all or args.attack:
        for comb in combs:
            for attack in args.attacks:
                if comb.dataset not in CODE_GENERATION_DATASETS and attack == "ours_no_logprobs": # We do not need to run our attack without logprobs for non-code generation tasks
                    continue
                if comb.dataset in CODE_GENERATION_DATASETS and attack not in {"ours", "ours_no_logprobs", "ours_trained"}: # Only our attacks are supported for code generation tasks
                    continue
                machine.add_task(AttackTask(machine, comb, attack))
        machine.dispatch_tasks()

    if args.process_results:
        paper_results_dir = os.path.join(RESULTS_DIR, "paper")
        os.makedirs(paper_results_dir, exist_ok=True)
        results = collect_results(combs, args.attacks)
        for dataset in args.datasets:
            processed = process_results(results, dataset)
            with open(os.path.join(paper_results_dir, f"{dataset}_results.{'tex' if args.latex else 'txt'}"), "w") as f:
                f.write(fmt_asr_qr_table(processed, dataset, args.latex))
                if not args.latex:
                    f.write("\n\n")
                f.write(fmt_nv_np_cb_table(processed, dataset, args.latex))
                if not args.latex:
                    f.write("\n")


if __name__ == "__main__":
    with torch.no_grad():
        main(parse_args())
