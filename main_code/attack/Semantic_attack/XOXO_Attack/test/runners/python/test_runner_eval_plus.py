import os
import unittest


class TestCanonicalFunctionTestRunner(unittest.TestCase):
    @unittest.skipUnless(os.getenv("TEST_RUNNER", False), "Skipping runner tests")
    def test_alll_canonical_sequential(self):
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
        from learning_programs.datasets.code_generation import BROKEN_EXAMPLES
        from learning_programs.runners.python.runner import FunctionTestRunner

        tasks = [t for t in (get_human_eval_plus() | get_mbpp_plus()).values() if t["task_id"] not in BROKEN_EXAMPLES]

        for task in tasks:
            with self.subTest(task_id=task["task_id"]):
                runner = FunctionTestRunner(task)
                res = runner.test(task["prompt"] + task["canonical_solution"])
                self.assertEqual(res.passed, True)
                self.assertAlmostEqual(res.share_tests_passed, 1.0)

    @unittest.skipUnless(os.getenv("TEST_RUNNER", False), "Skipping runner tests")
    def test_all_canonical(self):
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
        from learning_programs.datasets.code_generation import BROKEN_EXAMPLES
        from learning_programs.runners.python.runner import TestRunnerManager

        tasks = [t for t in (get_human_eval_plus() | get_mbpp_plus()).values() if t["task_id"] not in BROKEN_EXAMPLES]

        with TestRunnerManager() as mgr:
            results = mgr.test(tasks, [[t["prompt"] + t["canonical_solution"]] for t in tasks])

        for task, outputs in zip(tasks, results):
            with self.subTest(task_id=task["task_id"]):
                self.assertEqual(len(outputs), 1)
                outputs = outputs[0]
                self.assertEqual(outputs.passed, True)
                self.assertAlmostEqual(outputs.share_tests_passed, 1.0)
