import os
import pickle
import json
import subprocess
import tempfile

import multiprocessing
import threading
import warnings
from typing import Any, NamedTuple

import psutil
from evalplus.eval import PASS, TIMEOUT
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec
from tqdm import tqdm

from learning_programs.runners.python.evalplus_check import untrusted_check


SA_MAGIC_PARSER_ERROR = "StaticAnalyzerParseError"
SA_NO_ENTRY_POINT = "StaticAnalyzerNoEntryPoint"

DEFAULT_MIN_TIME_LIMIT = 1.0
DEFAULT_GT_TIME_LIMIT_FACTOR = 40.0

def _MB(x: int) -> int:
    """Converts x from MB to bytes."""
    return x * 1024**2


MAX_MEM_LIMIT = _MB(4096)  # 4 GB
RESERVED_MEM = _MB(4096)  # 4 GB reserved for system processes


class TestResult(NamedTuple):
    stat: str
    share_tests_passed: float

    @property
    def passed(self) -> bool:
        return self.stat == PASS

    @property
    def timed_out(self) -> bool:
        return self.stat == TIMEOUT


class FunctionTestRunner:
    """A function test runner. Used for running tests on functions."""

    task_id: str
    dataset: str
    entry_point: str
    tests_base: list[Any]
    tests_plus: list[Any]
    expected_plus: list[Any]
    expected_time_plus: float
    atol: float

    def __init__(self, problem: dict):
        self.task_id = problem["task_id"]
        self.dataset = "mbpp" if self.task_id.startswith("Mbpp") else "humaneval"
        self.entry_point = problem["entry_point"]
        self.tests_base = problem["base_input"]
        self.tests_plus = problem["plus_input"]
        self.atol = problem["atol"]
        output_not_none = self.entry_point in MBPP_OUTPUT_NOT_NONE_TASKS if self.task_id.startswith("Mbpp") else []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self.expected_plus, self.expected_time_plus = trusted_exec(
                problem["prompt"] + problem["canonical_solution"],
                problem["plus_input"],
                problem["entry_point"],
                record_time=True,
                output_not_none=output_not_none,
            )

    def test(self, code: str) -> TestResult:
        """Run the tests. Returns the share of passed tests."""

        # Errors caught by static analysis (e.g. syntax errors) are considered as failed tests
        if code.startswith(SA_MAGIC_PARSER_ERROR):
            return TestResult(SA_MAGIC_PARSER_ERROR, 0.0)
        if code.startswith(SA_NO_ENTRY_POINT):
            return TestResult(SA_NO_ENTRY_POINT, 0.0)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
        
            stat_plus, details_plus = untrusted_check(
                self.dataset,
                code,
                self.tests_plus,
                self.entry_point,
                self.expected_plus,
                self.atol,
                self.expected_time_plus,
                DEFAULT_MIN_TIME_LIMIT,
                DEFAULT_GT_TIME_LIMIT_FACTOR,
            )

        share_passed = sum(details_plus) / len(self.expected_plus)
        return TestResult(stat_plus, share_passed)


def get_max_mem_slots() -> int:
    usable_mem = psutil.virtual_memory().available - RESERVED_MEM
    if usable_mem < MAX_MEM_LIMIT:
        raise MemoryError(
            (
                f"Detected {_MB(usable_mem)} MB of available memory, which is less "
                f"than the maximum memory limit of {_MB(MAX_MEM_LIMIT)} MB."
                "Adjust the MAX_MEM_LIMIT accordingly."
            )
        )
    return usable_mem // MAX_MEM_LIMIT


def runners_cache_path(workdir: str) -> str:
    return os.path.join(workdir, "runners_cache.pkl")


def worker_assignments_path(workdir: str, worker_id: int) -> str:
    return os.path.join(workdir, f"assignments_{worker_id}.json")


class TestRunnerManager:
    """A test runner manager."""

    runners: dict[str, FunctionTestRunner]
    mem_sema: threading.Semaphore
    cpu_sema: threading.Semaphore
    max_workers: int
    max_mem_slots: int
    max_cpu_slots: int
    workdir: tempfile.TemporaryDirectory

    def __init__(self, max_workers: int = multiprocessing.cpu_count()):
        self.runners = {}
        self.max_workers = max_workers
        self.adjust_mem_resources()
        self.set_cpu_resources()

    def adjust_mem_resources(self):
        """Adjust the number of memory slots to reflect available memory."""
        self.max_mem_slots = get_max_mem_slots()
        self.mem_sema = threading.Semaphore(self.max_mem_slots)

    def set_cpu_resources(self):
        """Set the number of CPU resources to the number of available CPUs."""
        self.max_cpu_slots = multiprocessing.cpu_count()
        self.cpu_sema = threading.Semaphore(self.max_cpu_slots)

    def resource_acquire(self):
        """Acquire resources for a task."""
        self.mem_sema.acquire()
        self.cpu_sema.acquire()

    def resource_release(self):
        """Release resources for a task."""
        self.cpu_sema.release()
        self.mem_sema.release()

    def sema_wait(self):
        """Wait for all workers to finish."""
        for _ in range(self.max_mem_slots):
            self.mem_sema.acquire()
        for _ in range(self.max_cpu_slots):
            self.cpu_sema.acquire()
        self.cpu_sema.release(self.max_cpu_slots)
        self.mem_sema.release(self.max_mem_slots)

    def num_workers(self) -> int:
        return min(self.max_cpu_slots, self.max_mem_slots, self.max_workers)

    def prepare_runners(self, tasks: list[dict]):
        """Prepare runners for the tasks."""
        with tqdm(total=len(tasks)) as pbar:
            pbar.set_description("Preparing runners")
            for task in tasks:
                task_id = task["task_id"]
                if task_id not in self.runners:
                    self.runners[task_id] = FunctionTestRunner(task)
                pbar.update()

    def runners_cache_path(self) -> str:
        return runners_cache_path(self.workdir.name)

    def cache_runners(self):
        with open(self.runners_cache_path(), "wb") as f:
            pickle.dump(self.runners, f)

    def worker_assignments_path(self, worker_id: int) -> str:
        return worker_assignments_path(self.workdir.name, worker_id)

    def worker_cmd(self, worker_id: int) -> list[str]:
        return ["python", "-m", "learning_programs.runners.python.worker", self.workdir.name, str(worker_id)]

    def test(
        self, tasks: list[dict], generations: list[list[str]]
    ) -> list[list[TestResult]]:
        """Run the tests. Returns the share of passed tests."""

        if self.workdir is None:
            raise RuntimeError("TestRunnerManager must be used as a context manager.")

        self.adjust_mem_resources()
        self.prepare_runners(tasks)
        self.cache_runners()

        num_workers = self.num_workers()
        ret = [None] * len(generations)

        with tqdm(total=len(generations)) as pbar:
            pbar.set_description("Running tests")

            def wait_for_results(i: int, places: list[int]) -> None:
                try:
                    proc = subprocess.run(self.worker_cmd(i), check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(e.stderr)
                    raise
                for line, place in zip(proc.stdout.split("\n"), places):
                    ret[place] = [TestResult(*r) for r in json.loads(line)]
                    pbar.update(1)
                self.resource_release()

            task_ids = [t["task_id"] for t in tasks]
            for i in range(num_workers):
                self.resource_acquire()
                with open(self.worker_assignments_path(i), "w") as f:
                    json.dump(list(zip(task_ids[i::num_workers], generations[i::num_workers])), f)
                threading.Thread(target=wait_for_results, args=(i, list(range(i, len(generations), num_workers)))).start()

            self.sema_wait()
            return ret

    def __enter__(self):
        self.workdir = tempfile.TemporaryDirectory()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.workdir.cleanup()
