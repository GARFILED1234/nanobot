
import asyncio
import tempfile
import unittest
from pathlib import Path

from nanobot.planner.executor import DAGExecutor
from nanobot.planner.graph import TaskGraph
from nanobot.planner.models import (
    CriterionResult,
    Task,
    TaskStatus,
    ValidationResult,
    ValidationStatus,
)
from nanobot.planner.validation import TaskResultValidator


class StubValidator(TaskResultValidator):
    def __init__(self):
        self.calls = 0

    def validate(self, acceptance_criteria, workspace_root):
        self.calls += 1
        return ValidationResult(
            passed=False,
            feedback=["文件验收失败"],
            criterion_results=[
                CriterionResult(
                    criterion_id="C1",
                    passed=False,
                    feedback=["目标文件不存在"],
                )
            ],
        )


class TestDAGExecutorRetryState(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workspace_root = Path(self.temp_dir.name)

    async def test_retry_start_resets_previous_validation_state(self):
        graph = TaskGraph()
        task = Task("T1", "需要重试的任务", max_retries=2)
        graph.add_task(task)
        validator = StubValidator()
        execution_count = 0
        retry_started = asyncio.Event()
        release_retry = asyncio.Event()

        async def execute_task(current_task):
            nonlocal execution_count
            execution_count += 1
            if execution_count == 1:
                return "第一次执行结果"
            retry_started.set()
            await release_retry.wait()
            return "第二次执行结果"

        executor = DAGExecutor(
            graph=graph,
            execute_task=execute_task,
            validator=validator,
            workspace_root=self.workspace_root,
        )
        with self.assertRaisesRegex(RuntimeError, "任务验收失败"):
            await executor._execute_one(task)
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertEqual(task.validation_status, ValidationStatus.FAILED)
        self.assertIsNotNone(task.validation_result)

        retry_task = asyncio.create_task(executor._execute_one(task))
        try:
            await asyncio.wait_for(retry_started.wait(), timeout=2)
            self.assertEqual(task.status, TaskStatus.RUNNING)
            self.assertEqual(task.validation_status, ValidationStatus.NOT_CHECKED)
            self.assertIsNone(task.validation_result)
        finally:
            release_retry.set()
            await asyncio.gather(retry_task, return_exceptions=True)

    async def test_validation_failure_exhaustion_records_error(self):
        graph = TaskGraph()
        task = Task("T1", "验收始终失败的任务", max_retries=1)
        graph.add_task(task)
        validator = StubValidator()
        execution_count = 0

        async def execute_task(current_task):
            nonlocal execution_count
            execution_count += 1
            return f"第{execution_count}次执行结果"

        executor = DAGExecutor(
            graph=graph,
            execute_task=execute_task,
            validator=validator,
            workspace_root=self.workspace_root,
        )
        await executor.execute()
        self.assertEqual(execution_count, 2)
        self.assertEqual(validator.calls, 2)
        self.assertEqual(task.retry_count, 1)
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertEqual(task.validation_status, ValidationStatus.FAILED)
        self.assertIsNotNone(task.error)
        self.assertIn("任务验收失败", task.error)
        self.assertEqual(task.validation_result.feedback, ["文件验收失败"])

    async def test_retry_success_clears_previous_error(self):
        graph = TaskGraph()
        task = Task("T1", "重试后成功的任务", max_retries=1)
        graph.add_task(task)
        failed_result = ValidationResult(
            passed=False,
            feedback=["第一次验收失败"],
            criterion_results=[
                CriterionResult(
                    criterion_id="C1",
                    passed=False,
                    feedback=["文件不存在"],
                )
            ],
        )
        passed_result = ValidationResult(
            passed=True,
            feedback=["第二次验收通过"],
            criterion_results=[
                CriterionResult(
                    criterion_id="C1",
                    passed=True,
                    feedback=["文件检查通过"],
                )
            ],
        )

        class SequenceValidator(TaskResultValidator):
            def __init__(self):
                self.results = [failed_result, passed_result]

            def validate(self, acceptance_criteria, workspace_root):
                return self.results.pop(0)

        execution_count = 0

        async def execute_task(current_task):
            nonlocal execution_count
            execution_count += 1
            return f"第{execution_count}次执行结果"

        executor = DAGExecutor(
            graph=graph,
            execute_task=execute_task,
            validator=SequenceValidator(),
            workspace_root=self.workspace_root,
        )
        await executor.execute()
        self.assertEqual(execution_count, 2)
        self.assertEqual(task.retry_count, 1)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.validation_status, ValidationStatus.PASSED)
        self.assertTrue(task.validation_result.passed)
        self.assertIsNone(task.error)
        self.assertEqual(task.result, "第2次执行结果")


if __name__ == "__main__":
    unittest.main()
