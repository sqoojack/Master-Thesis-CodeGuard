import multiprocessing
import os
import time
from typing import Any, Optional

import numpy as np
import psutil

from evalplus.eval._special_oracle import (
    MBPP_OUTPUT_NOT_NONE_TASKS,
    MBPP_OUTPUT_SET_EQ_TASKS,
    _digit_distance_nums,
    _poly,
    _surface_Area,
)
from evalplus.eval.utils import (
    reliability_guard,
    swallow_io,
    time_limit,
)

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


def query_maximum_memory_bytes() -> Optional[int]:
    # Disable functionalities that can make destructive changes to the test.
    # allow only 4GB memory usage
    maximum_memory_bytes = os.getenv(
        "EVALPLUS_MAX_MEMORY_BYTES", 4 * 1024 * 1024 * 1024
    )
    maximum_memory_bytes = min(int(maximum_memory_bytes), psutil.virtual_memory().total)
    if maximum_memory_bytes == -1:
        return None
    return maximum_memory_bytes


def is_floats(x) -> bool:
    # check if it is float; List[float]; Tuple[float]
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)) and x:
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False


def check_output(inp: Any, out: Any, exp: Any, atol: float, dataset: str, entry_point: str) -> bool:
    exact_match = out == exp

    # ================================================ #
    # ============== special oracles ================= #
    if dataset == "mbpp":
        if "are_equivalent" == entry_point:  # Mbpp/164 special oracle
            exact_match = exact_match or True
        elif "sum_div" == entry_point:  # Mbpp/295 special oracle
            exact_match = exact_match or out == 0
        elif "surface_Area" == entry_point:  # Mbpp/581 special oracle
            exact_match = (
                exact_match or abs(out - _surface_Area(*inp)) <= atol
            )
        elif (
            "digit_distance_nums" == entry_point
        ):  # Mbpp/558 special oracle
            exact_match = exact_match or out == _digit_distance_nums(
                *inp
            )
        elif entry_point in MBPP_OUTPUT_SET_EQ_TASKS:
            exact_match = set(out) == set(exp)
        elif entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
            # exp is True  if not None
            #        False if None
            if isinstance(out, bool):
                exact_match = out == exp
            else:
                exact_match = exp == (out is not None)

    if dataset == "humaneval":
        if "find_zero" == entry_point:
            return abs(_poly(*inp, out)) <= atol

    # ============== special oracles ================= #
    # ================================================ #

    if atol == 0 and is_floats(exp):
        atol = 1e-6  # enforce atol for float comparison
    if not exact_match and atol != 0:
        # explicitly set rtol=1e-07
        # to match `np.testing.assert_allclose`'s default values
        if type(out) is not type(exp):
            return False
        if isinstance(exp, (list, tuple)):
            if len(out) != len(exp):
                return False
        return np.allclose(out, exp, rtol=1e-07, atol=atol)

    return exact_match


def unsafe_execute(
    dataset: str,
    entry_point: str,
    code: str,
    inputs,
    expected: list,
    time_limits,
    atol,
    lock, # Lock
    stat,  # Value
    details,  # Array
    progress,  # Value
):
    lock.acquire()
    reliability_guard(maximum_memory_bytes=query_maximum_memory_bytes())
    exec_globals = {}
    try:
        with swallow_io():
            exec(code, exec_globals)
            fn = exec_globals[entry_point]

        for i, inp in enumerate(inputs):
            try:
                with time_limit(time_limits[i]):
                    with swallow_io():
                        out = fn(*inp)
                    
                exp = expected[i]
                details[i] = check_output(inp, out, exp, atol, dataset, entry_point)
                progress.value += 1

            except BaseException:
                details[i] = False
                progress.value += 1
                continue

        stat.value = _SUCCESS
    except BaseException:
        stat.value = _FAILED
    lock.release()


def untrusted_check(
    dataset: str,
    code: str,
    inputs: list[Any],
    entry_point: str,
    expected,
    atol,
    ref_time: list[float],
    min_time_limit: float,
    gt_time_limit_factor: float,
) -> tuple[str, np.ndarray]:
    time_limits = [max(min_time_limit, gt_time_limit_factor * t) for t in ref_time]
    timeout = min(os.getenv("EVALPLUS_TIMEOUT_PER_TASK", 60), sum(time_limits)) + 2

    # shared memory objects
    lock = multiprocessing.Lock()
    progress = multiprocessing.Value("i", 0, lock=False)
    stat = multiprocessing.Value("i", _UNKNOWN, lock=False)
    details = multiprocessing.Array("b", [False for _ in range(len(inputs))], lock=False)

    p = multiprocessing.Process(
        target=unsafe_execute,
        args=(
            dataset,
            entry_point,
            code,
            inputs,
            expected,
            time_limits,
            atol,
            # return values
            lock,
            stat,
            details,
            progress,
        ),
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)
    
    if lock.acquire(block=False):
        lock.release()
    else:
        return TIMEOUT, []

    stat = _mapping[stat.value]
    details = details[: progress.value]

    if not stat:
        stat = TIMEOUT

    if stat == PASS:
        if len(details) != len(inputs) or not all(details):
            stat = FAIL

    return stat, details
