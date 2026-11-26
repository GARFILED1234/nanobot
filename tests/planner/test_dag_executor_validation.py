
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
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = 0

    def validate(self, acceptance_criteria, workspace_root):
        self.calls += 1
        if self.results:
            return self.results.pop(0)
        return ValidationResult(passed=True)


class StalledTaskGraph(TaskGraph):
    def mark_blocked_tasks(self):
        return []

    def get_ready_tasks(self):
        return False, []


class TestDAGExecutor(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_executor(
        self,
        graph,
        execute_task,
        validator=None,
        max_concurrency=3,
    ):
        return DAGExecutor(
            graph=graph,
            execute_task=execute_task,
            validator=validator or StubValidator(),
            workspace_root=self.workspace_root,
            max_concurrency=max_concurrency,
        )

    async def test_normal_execution_and_dependency_order(self):
        graph = TaskGraph()
        t1 = Task("T1", "任务一")
        t2 = Task("T2", "任务二")
        t3 = Task("T3", "任务三")
        for task in (t1, t2, t3):
            graph.add_task(task)
        graph.add_dependency("T2", "T1")
        graph.add_dependency("T3", "T2")
        execution_order = []

        async def execute_task(task):
            execution_order.append(f"{task.task_id}_START")
            if task.task_id == "T2":
                self.assertEqual(t1.status, TaskStatus.COMPLETED)
            if task.task_id == "T3":
                self.assertEqual(t2.status, TaskStatus.COMPLETED)
            execution_order.append(f"{task.task_id}_END")
            return f"{task.task_id}_result"

        executor = self.create_executor(graph, execute_task)
        await executor.execute()

        self.assertEqual(t1.status, TaskStatus.COMPLETED)
        self.assertEqual(t2.status, TaskStatus.COMPLETED)
        self.assertEqual(t3.status, TaskStatus.COMPLETED)
        self.assertEqual(t1.result, "T1_result")
        self.assertEqual(t2.result, "T2_result")
        self.assertEqual(t3.result, "T3_result")
        self.assertLess(
            execution_order.index("T1_END"),
            execution_order.index("T2_START"),
        )
        self.assertLess(
            execution_order.index("T2_END"),
            execution_order.index("T3_START"),
        )

    async def test_max_concurrency_limit(self):
        graph = TaskGraph()
        tasks = [
            Task(f"T{i}", f"任务{i}")
            for i in range(5)
        ]
        for task in tasks:
            graph.add_task(task)

        active_count = 0
        max_active_count = 0
        started_count = 0
        two_started = asyncio.Event()
        release_tasks = asyncio.Event()

        async def execute_task(task):
            nonlocal active_count
            nonlocal max_active_count
            nonlocal started_count
            active_count += 1
            started_count += 1
            max_active_count = max(max_active_count, active_count)
            if started_count >= 2:
                two_started.set()
            try:
                await release_tasks.wait()
                return f"{task.task_id}_result"
            finally:
                active_count -= 1

        executor = self.create_executor(
            graph,
            execute_task,
            max_concurrency=2,
        )
        execution = asyncio.create_task(executor.execute())
        await asyncio.wait_for(two_started.wait(), timeout=2)
        self.assertEqual(active_count, 2)
        self.assertEqual(max_active_count, 2)
        release_tasks.set()
        await asyncio.wait_for(execution, timeout=2)

        self.assertEqual(max_active_count, 2)
        for task in tasks:
            self.assertEqual(task.status, TaskStatus.COMPLETED)

    async def test_execution_exception_retries_until_success(self):
        graph = TaskGraph()
        task = Task(
            "T1",
            "需要重试的任务",
            max_retries=2,
        )
        graph.add_task(task)
        execution_count = 0

        async def execute_task(current_task):
            nonlocal execution_count
            execution_count += 1
            if execution_count < 3:
                raise RuntimeError("模拟执行异常")
            return "执行成功"

        executor = self.create_executor(graph, execute_task)
        await executor.execute()

        self.assertEqual(execution_count, 3)
        self.assertEqual(task.retry_count, 2)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.result, "执行成功")
        self.assertIsNone(task.error)
        self.assertEqual(
            task.validation_status,
            ValidationStatus.PASSED,
        )

    async def test_max_retries_exhausted(self):
        graph = TaskGraph()
        task = Task(
            "T1",
            "持续失败的任务",
            max_retries=3,
        )
        graph.add_task(task)
        execution_count = 0

        async def execute_task(current_task):
            nonlocal execution_count
            execution_count += 1
            raise RuntimeError("持续执行失败")

        executor = self.create_executor(graph, execute_task)
        await executor.execute()

        self.assertEqual(execution_count, 4)
        self.assertEqual(task.retry_count, 3)
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertEqual(task.error, "持续执行失败")
        self.assertEqual(
            task.validation_status,
            ValidationStatus.NOT_CHECKED,
        )
        self.assertIsNone(task.validation_result)

    async def test_validation_failure_retries_and_then_passes(self):
        graph = TaskGraph()
        task = Task(
            "T1",
            "验收失败后修复的任务",
            max_retries=2,
        )
        graph.add_task(task)

        failed_result = ValidationResult(
            passed=False,
            feedback=["文件缺少必要内容"],
            criterion_results=[
                CriterionResult(
                    criterion_id="C1",
                    passed=False,
                    feedback=["缺少必要内容"],
                )
            ],
        )
        passed_result = ValidationResult(
            passed=True,
            feedback=["验收通过"],
            criterion_results=[
                CriterionResult(
                    criterion_id="C1",
                    passed=True,
                    feedback=["检查通过"],
                )
            ],
        )
        validator = StubValidator([failed_result, passed_result])
        execution_count = 0

        async def execute_task(current_task):
            nonlocal execution_count
            execution_count += 1
            return f"第{execution_count}次执行结果"

        executor = self.create_executor(
            graph,
            execute_task,
            validator=validator,
        )
        await executor.execute()

        self.assertEqual(execution_count, 2)
        self.assertEqual(validator.calls, 2)
        self.assertEqual(task.retry_count, 1)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.result, "第2次执行结果")
        self.assertEqual(
            task.validation_status,
            ValidationStatus.PASSED,
        )
        self.assertTrue(task.validation_result.passed)

    async def test_validation_failure_exhausted_preserves_feedback(self):
        graph = TaskGraph()
        task = Task(
            "T1",
            "始终无法通过验收的任务",
            max_retries=1,
        )
        graph.add_task(task)

        failed_result = ValidationResult(
            passed=False,
            feedback=["验收没有通过"],
            criterion_results=[
                CriterionResult(
                    criterion_id="C1",
                    passed=False,
                    feedback=["目标文件不存在"],
                )
            ],
        )
        validator = StubValidator([
            failed_result,
            failed_result,
        ])
        execution_count = 0

        async def execute_task(current_task):
            nonlocal execution_count
            execution_count += 1
            return "Subagent 已返回"

        executor = self.create_executor(
            graph,
            execute_task,
            validator=validator,
        )
        await executor.execute()

        self.assertEqual(execution_count, 2)
        self.assertEqual(task.retry_count, 1)
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertEqual(
            task.validation_status,
            ValidationStatus.FAILED,
        )
        self.assertIs(task.validation_result, failed_result)
        self.assertEqual(
            task.validation_result.feedback,
            ["验收没有通过"],
        )

    async def test_failed_task_blocks_dependents_but_not_independent_tasks(self):
        graph = TaskGraph()
        t1 = Task(
            "T1",
            "失败任务",
            max_retries=0,
        )
        t2 = Task("T2", "依赖T1的任务")
        t3 = Task("T3", "独立任务")
        t4 = Task("T4", "依赖T2的任务")
        for task in (t1, t2, t3, t4):
            graph.add_task(task)
        graph.add_dependency("T2", "T1")
        graph.add_dependency("T4", "T2")
        execution_count = {
            "T1": 0,
            "T2": 0,
            "T3": 0,
            "T4": 0,
        }

        async def execute_task(task):
            execution_count[task.task_id] += 1
            if task.task_id == "T1":
                raise RuntimeError("T1执行失败")
            return f"{task.task_id}_result"

        executor = self.create_executor(graph, execute_task)
        await executor.execute()

        self.assertEqual(t1.status, TaskStatus.FAILED)
        self.assertEqual(t2.status, TaskStatus.BLOCKED)
        self.assertEqual(t4.status, TaskStatus.BLOCKED)
        self.assertEqual(t3.status, TaskStatus.COMPLETED)
        self.assertEqual(execution_count["T1"], 1)
        self.assertEqual(execution_count["T2"], 0)
        self.assertEqual(execution_count["T3"], 1)
        self.assertEqual(execution_count["T4"], 0)

    async def test_running_dependent_task_is_cancelled_and_blocked(self):
        graph = TaskGraph()
        t1 = Task("T1", "已经失败的前置任务")
        t2 = Task("T2", "正在运行的下游任务")
        t3 = Task("T3", "更下游的任务")
        for task in (t1, t2, t3):
            graph.add_task(task)
        graph.add_dependency("T2", "T1")
        graph.add_dependency("T3", "T2")
        t1.status = TaskStatus.FAILED
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def execute_task(task):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        executor = self.create_executor(graph, execute_task)
        running_task = asyncio.create_task(executor._execute_one(t2))
        executor.running_tasks[t2.task_id] = running_task
        await asyncio.wait_for(started.wait(), timeout=2)

        await asyncio.wait_for(executor.execute(), timeout=2)

        self.assertTrue(cancelled.is_set())
        self.assertTrue(running_task.done())
        self.assertEqual(t2.status, TaskStatus.BLOCKED)
        self.assertEqual(t3.status, TaskStatus.BLOCKED)
        self.assertNotIn("T2", executor.running_tasks)

    async def test_cancel_task_handles_already_failed_coroutine(self):
        graph = TaskGraph()
        task = Task("T1", "已经异常结束的协程")
        graph.add_task(task)

        async def execute_task(current_task):
            raise RuntimeError("协程提前失败")

        executor = self.create_executor(graph, execute_task)
        running_task = asyncio.create_task(executor._execute_one(task))
        executor.running_tasks[task.task_id] = running_task

        with self.assertRaisesRegex(RuntimeError, "协程提前失败"):
            await running_task

        await executor.cancel_task(task)

        self.assertEqual(task.status, TaskStatus.BLOCKED)
        self.assertEqual(task.error, "协程提前失败")

    async def test_scheduler_stall_raises_runtime_error(self):
        graph = StalledTaskGraph()

        async def execute_task(task):
            return "不会执行"

        executor = self.create_executor(graph, execute_task)

        with self.assertRaisesRegex(
            RuntimeError,
            "DAG 调度停滞",
        ):
            await executor.execute()


if __name__ == "__main__":
    unittest.main()
