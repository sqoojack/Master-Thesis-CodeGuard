import json
import os
import pickle
import tempfile
import sys

from learning_programs.runners.python.runner import FunctionTestRunner, runners_cache_path, worker_assignments_path


def checker_worker(workdir: str, worker_id: int):
    """Worker for checking the results of the assignments."""

    load_path = worker_assignments_path(workdir, worker_id)

    with open(load_path, "r") as f:
        assignments: list[str, list[str]] = json.load(f)

    with open(runners_cache_path(workdir), "rb") as f:
        runners: dict[str, FunctionTestRunner] = pickle.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        for task, generations in assignments:
            res = [runners[task].test(g) for g in generations]
            print(json.dumps(res))


if __name__ == "__main__":
    checker_worker(sys.argv[1], int(sys.argv[2]))
