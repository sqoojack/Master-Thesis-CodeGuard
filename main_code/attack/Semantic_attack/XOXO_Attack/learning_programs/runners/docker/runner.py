import json
import os
import subprocess
import tempfile
import uuid
from typing import NamedTuple
from learning_programs.runners.python.runner import SA_NO_ENTRY_POINT, SA_MAGIC_PARSER_ERROR

DATA_PATH = "learning_programs/datasets/CWEval/benchmark/core/py"


class TestResult(NamedTuple):
    functional: bool
    secure: bool

    @property
    def passed(self) -> bool:
        return not (self.functional and not self.secure)

    @property
    def timed_out(self) -> bool:
        return False

    @property
    def share_tests_passed(self) -> float:
        return 1.0 if self.passed else 0.0


def cmd_exec(cmd: str, err_msg: str) -> str:
    try:
        return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{err_msg}: {e.stderr}") from e

class TestRunnerManager:
    IMAGE: str = "pytest-ubuntu"
    CONTAINER: str = "pytest-ubuntu"
    RESULTS: str = "results.json"
    TEST_DIR: str = "test"
    workdir: tempfile.TemporaryDirectory
    test_files: dict[str, str]
    WORKSPACE: str = "/home/ubuntu"
    max_workers: int
    results: list[TestResult]

    def __init__(self, max_workers: int):
        cmd_exec(["docker", "build", "-t", self.IMAGE, "learning_programs/runners/docker"], "Failed to build the docker image")
        self.CONTAINER = f"{self.CONTAINER}_{str(uuid.uuid4())[:8]}"
        self.test_files = self.load_tests()
        self.max_workers = max_workers

    def __enter__(self):
        cmd_exec(["docker", "run", "--cpus", str(self.max_workers), "--rm", "-d", "--name", self.CONTAINER, self.IMAGE], "Failed to start the docker container")
        self.workdir = tempfile.TemporaryDirectory()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        cmd_exec(["docker", "stop", self.CONTAINER], "Failed to stop the docker container")
        self.workdir.cleanup()

    def load_tests(self):
        test_files = dict()
        for dirent in os.scandir(DATA_PATH):
            if dirent.name.endswith("_test.py"):
                module_name = dirent.name.replace("_test.py", "_task")
                with open(dirent.path, "r") as f:
                    file = f.read()
                test_files[module_name] = file
        return test_files

    def setup_codes(self, tasks: list[dict], generations: list[list[str]]):
        self.results = [[TestResult(False, False)] for _ in generations]
        for i, (generated_codes, task) in enumerate(zip(generations, tasks)):
            generation = generated_codes[0]
            if generation.startswith(SA_NO_ENTRY_POINT) or generation.startswith(SA_MAGIC_PARSER_ERROR):
                continue
            name = task["task_id"].split("_", 1)[1]
            module_name = f"{name}_task"
            new_module_name = f"{name}_{i}_task"
            with open(f"{self.workdir.name}/{new_module_name}.py", "w") as f:
                f.write(generation)

            test_file = self.test_files[module_name].replace(f"from {module_name} ", f"from {new_module_name} ", 1)
            new_test_name = f"{name}_{i}_test"
            with open(f"{self.workdir.name}/{new_test_name}.py", "w") as f:
                f.write(test_file)

    def copy_codes_to_container(self):
        cmd_exec(["docker", "cp", f"{self.workdir.name}/", f"{self.CONTAINER}:{self.WORKSPACE}/test"], "Failed to copy the codes to the container")
        cmd_exec(["docker", "exec", "-u", "root", "-w", self.WORKSPACE, self.CONTAINER, "chmod", "-R", "755", self.TEST_DIR], "Failed to add permissions to the test directory")

    def clean_up_workdir(self):
        for dirent in os.scandir(self.workdir.name):
            os.remove(dirent.path)

    def run_tests(self):
        cmd_exec(["docker", "exec", "-w", self.WORKSPACE, self.CONTAINER, "bash", "run_tests.sh"], "Failed to run the tests")

    def extract_results_from_container(self):
        results = json.loads(cmd_exec(["docker", "exec", self.CONTAINER, "cat", f"{self.WORKSPACE}/{self.RESULTS}"], "Failed to extract the results"))
        for result in results:
            self.results[int(result["file"].split("_")[-1])] = [TestResult(result["functional"], result["secure"])]

    def clean_up_container(self):
        cmd_exec(["docker", "exec", "-u", "root", "-w", self.WORKSPACE, self.CONTAINER, "rm", "-rf", self.TEST_DIR, self.RESULTS], "Failed to clean up the container")

    def test(self, tasks: list[dict], generations: list[list[str]]) -> list[TestResult]:
        if self.workdir is None:
            raise RuntimeError("TestRunnerManager must be used as a context manager.")
        self.setup_codes(tasks, generations)
        self.copy_codes_to_container()
        self.clean_up_workdir()
        self.run_tests()
        self.extract_results_from_container()
        self.clean_up_container()
        return self.results


def load_solutions():
    test_files = dict()
    for dirent in os.scandir(DATA_PATH):
        if dirent.name.endswith("_task.py"):
            module_name = dirent.name.replace("_task.py", "")
            with open(dirent.path, "r") as f:
                file = f.read()
            test_files[module_name] = file
    return test_files


if __name__ == "__main__":
    solutions = load_solutions()
    tasks = [{"task_id": f"CWEval/py_{task_id}"} for task_id in solutions.keys()]
    generations = [[code] for code in solutions.values()]
    with TestRunnerManager(10) as t:
        results = t.test(tasks, generations)
        for result in results:
            print(result)
        results = t.test(tasks, generations)
        for result in results:
            print(result)
        results = t.test(tasks, generations)
        for result in results:
            print(result)