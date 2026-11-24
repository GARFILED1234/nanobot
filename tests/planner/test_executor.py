
import unittest
import asyncio

from nanobot.planner.models import Task, TaskStatus
from nanobot.planner.graph import TaskGraph
from nanobot.planner.executor import DAGExecutor


class TestDAGExecutor(unittest.IsolatedAsyncioTestCase):

    async def test_concurrent_execution(self):
        graph = TaskGraph()

        # 创建三个任务
        t1 = Task(task_id="T1", description="任务1")
        t2 = Task(task_id="T2", description="任务2")
        t3 = Task(task_id="T3", description="任务3")

        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)

        # T3 依赖 T1 和 T2
        graph.add_dependency("T3", "T1")
        graph.add_dependency("T3", "T2")

        # 记录任务执行顺序
        execution_order = []

        # 用于检测 T1 和 T2 是否并发执行
        started = set()

        async def execute_task(task):
            execution_order.append(f"{task.task_id}_START")
            started.add(task.task_id)

            # T1、T2 必须同时进入执行函数
            if task.task_id in ("T1", "T2"):
                await asyncio.sleep(0.1)

            if task.task_id == "T3":
                # T3 执行时，T1 和 T2 必须已经完成
                self.assertEqual(
                    graph.get_task("T1").status,
                    TaskStatus.COMPLETED,
                )
                self.assertEqual(
                    graph.get_task("T2").status,
                    TaskStatus.COMPLETED,
                )

            execution_order.append(f"{task.task_id}_END")
            return f"{task.task_id}_result"

        executor = DAGExecutor(
            graph=graph,
            execute_task=execute_task,
            max_concurrency=2,
        )

        await executor.execute()

        # 检查所有任务最终完成
        self.assertEqual(t1.status, TaskStatus.COMPLETED)
        self.assertEqual(t2.status, TaskStatus.COMPLETED)
        self.assertEqual(t3.status, TaskStatus.COMPLETED)

        # 检查结果
        self.assertEqual(t1.result, "T1_result")
        self.assertEqual(t2.result, "T2_result")
        self.assertEqual(t3.result, "T3_result")

        # 检查 T1、T2 都执行过
        self.assertIn("T1_START", execution_order)
        self.assertIn("T2_START", execution_order)

        # 检查 T3 在 T1、T2 完成后才开始
        self.assertLess(
            execution_order.index("T1_END"),
            execution_order.index("T3_START"),
        )
        self.assertLess(
            execution_order.index("T2_END"),
            execution_order.index("T3_START"),
        )

    
    async def test_retry_and_dependency_execution(self):
        graph = TaskGraph()

        t1 = Task(task_id="T1", description="任务1")
        t2 = Task(
            task_id="T2",
            description="任务2",
            max_retries=2,
        )
        t3 = Task(task_id="T3", description="任务3")

        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)

        graph.add_dependency("T2", "T1")
        graph.add_dependency("T3", "T2")

        execution_count = {
            "T1": 0,
            "T2": 0,
            "T3": 0,
        }

        async def execute_task(task):
            execution_count[task.task_id] += 1

            if task.task_id == "T2":
                # T2 第一次失败，第二次成功
                if execution_count["T2"] == 1:
                    raise RuntimeError("模拟任务失败")

            if task.task_id == "T3":
                # T3 执行时，T2 必须已经成功
                self.assertEqual(
                    t2.status,
                    TaskStatus.COMPLETED,
                )

            return f"{task.task_id}_result"

        executor = DAGExecutor(
            graph=graph,
            execute_task=execute_task,
            max_concurrency=2,
        )

        await executor.execute()

        # 检查 T2 确实执行了两次
        self.assertEqual(execution_count["T2"], 2)

        # 检查重试次数
        self.assertEqual(t2.retry_count, 1)

        # 检查三个任务最终完成
        self.assertEqual(t1.status, TaskStatus.COMPLETED)
        self.assertEqual(t2.status, TaskStatus.COMPLETED)
        self.assertEqual(t3.status, TaskStatus.COMPLETED)

        # 检查每个任务的执行次数
        self.assertEqual(execution_count["T1"], 1)
        self.assertEqual(execution_count["T3"], 1)


    async def test_retry_exhausted_blocks_dependents(self):
        graph = TaskGraph()

        t1 = Task(task_id="T1", description="任务1")
        t2 = Task(
            task_id="T2",
            description="任务2",
            max_retries=2,
        )
        t3 = Task(task_id="T3", description="任务3")

        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)

        graph.add_dependency("T2", "T1")
        graph.add_dependency("T3", "T2")

        execution_count = {
            "T1": 0,
            "T2": 0,
            "T3": 0,
        }

        async def execute_task(task):
            execution_count[task.task_id] += 1

            if task.task_id == "T2":
                raise RuntimeError("模拟持续失败")

            return f"{task.task_id}_result"

        executor = DAGExecutor(
            graph=graph,
            execute_task=execute_task,
            max_concurrency=2,
        )

        await executor.execute()

        self.assertEqual(execution_count["T1"], 1)
        self.assertEqual(execution_count["T2"], 3)
        self.assertEqual(execution_count["T3"], 0)

        self.assertEqual(t1.status, TaskStatus.COMPLETED)
        self.assertEqual(t2.status, TaskStatus.FAILED)
        self.assertEqual(t3.status, TaskStatus.BLOCKED)

        self.assertEqual(t2.retry_count, 2)

    
    async def test_max_concurrency_limit(self):
        graph = TaskGraph()

        tasks = [
            Task(task_id=f"T{i}", description=f"任务{i}")
            for i in range(1, 6)
        ]

        for task in tasks:
            graph.add_task(task)

        active_count = 0
        max_active_count = 0

        async def execute_task(task):
            nonlocal active_count, max_active_count

            active_count += 1
            max_active_count = max(
                max_active_count,
                active_count,
            )

            try:
                await asyncio.sleep(0.1)
            finally:
                active_count -= 1

            return f"{task.task_id}_result"

        executor = DAGExecutor(
            graph=graph,
            execute_task=execute_task,
            max_concurrency=2,
        )

        await executor.execute()

        # 检查最大并发数没有超过限制
        self.assertLessEqual(max_active_count, 2)

        # 确认确实发生了并发执行
        self.assertEqual(max_active_count, 2)

        # 检查所有任务最终完成
        for task in tasks:
            self.assertEqual(
                task.status,
                TaskStatus.COMPLETED,
            )


    async def test_failure_does_not_affect_independent_tasks(self):
        graph = TaskGraph()

        t1 = Task(task_id="T1", description="任务1")
        t2 = Task(
            task_id="T2",
            description="任务2",
            max_retries=0,
        )
        t3 = Task(task_id="T3", description="任务3")

        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)

        graph.add_dependency("T2", "T1")

        execution_count = {
            "T1": 0,
            "T2": 0,
            "T3": 0,
        }

        async def execute_task(task):
            execution_count[task.task_id] += 1

            if task.task_id == "T2":
                raise RuntimeError("模拟任务失败")

            return f"{task.task_id}_result"

        executor = DAGExecutor(
            graph=graph,
            execute_task=execute_task,
            max_concurrency=2,
        )

        await executor.execute()

        self.assertEqual(t1.status, TaskStatus.COMPLETED)
        self.assertEqual(t2.status, TaskStatus.FAILED)
        self.assertEqual(t3.status, TaskStatus.COMPLETED)

        self.assertEqual(execution_count["T1"], 1)
        self.assertEqual(execution_count["T2"], 1)
        self.assertEqual(execution_count["T3"], 1)


    async def test_cancel_running_blocked_task(self):
        graph = TaskGraph()

        t1 = Task(task_id="T1", description="任务1")
        t2 = Task(task_id="T2", description="任务2")
        t3 = Task(task_id="T3", description="任务3")

        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)

        graph.add_dependency("T2", "T1")
        graph.add_dependency("T3", "T2")

        # 模拟 T1 失败，T2 正在运行
        t1.status = TaskStatus.FAILED
        t2.status = TaskStatus.RUNNING

        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def execute_task(task):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        executor = DAGExecutor(
            graph=graph,
            execute_task=execute_task,
        )

        # 模拟一个已经运行的协程
        async_task = asyncio.create_task(
            executor._execute_one(t2)
        )
        executor.running_tasks[t2.task_id] = async_task

        await started.wait()

        # 获取需要取消的任务
        to_cancel = graph.mark_blocked_tasks()

        self.assertEqual(
            [task.task_id for task in to_cancel],
            ["T2"],
        )

        self.assertEqual(t3.status, TaskStatus.BLOCKED)

        # 取消 T2
        await executor.cancel_task(t2)

        self.assertTrue(cancelled.is_set())
        self.assertTrue(async_task.done())

        self.assertEqual(t2.status, TaskStatus.BLOCKED)


    async def test_cancelled_task_is_removed_from_running_tasks(self):
        graph = TaskGraph()

        t1 = Task(task_id="T1", description="任务1")
        t2 = Task(task_id="T2", description="任务2")

        graph.add_task(t1)
        graph.add_task(t2)

        graph.add_dependency("T2", "T1")

        # 模拟前置任务失败，下游任务正在运行
        t1.status = TaskStatus.FAILED
        t2.status = TaskStatus.RUNNING

        started = asyncio.Event()

        async def execute_task(task):
            started.set()
            await asyncio.Event().wait()

        executor = DAGExecutor(
            graph=graph,
            execute_task=execute_task,
        )

        # 模拟已经提交的运行中协程
        async_task = asyncio.create_task(
            executor._execute_one(t2)
        )
        executor.running_tasks[t2.task_id] = async_task

        await started.wait()

        # 通过正常的 Executor 调度流程处理取消
        await executor.execute()

        # 检查任务状态
        self.assertEqual(t2.status, TaskStatus.BLOCKED)

        # 检查协程确实退出
        self.assertTrue(async_task.done())

        # 检查运行任务映射是否清理
        self.assertNotIn(t2.task_id, executor.running_tasks)


    async def test_cancel_multiple_running_tasks(self):
        graph = TaskGraph()

        t1 = Task(task_id="T1", description="任务1")
        t2 = Task(task_id="T2", description="任务2")
        t3 = Task(task_id="T3", description="任务3")
        t4 = Task(task_id="T4", description="任务4")
        t5 = Task(task_id="T5", description="任务5")

        for task in [t1, t2, t3, t4, t5]:
            graph.add_task(task)

        graph.add_dependency("T2", "T1")
        graph.add_dependency("T3", "T1")
        graph.add_dependency("T4", "T2")
        graph.add_dependency("T5", "T3")

        # 模拟 T1 失败
        t1.status = TaskStatus.FAILED

        started = set()
        cancelled = set()
        both_started = asyncio.Event()

        async def execute_task(task):
            started.add(task.task_id)

            if len(started) == 2:
                both_started.set()

            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add(task.task_id)

        executor = DAGExecutor(
            graph=graph,
            execute_task=execute_task,
            max_concurrency=2,
        )

        # 模拟 T2、T3 已经运行
        task2 = asyncio.create_task(
            executor._execute_one(t2)
        )
        task3 = asyncio.create_task(
            executor._execute_one(t3)
        )

        executor.running_tasks["T2"] = task2
        executor.running_tasks["T3"] = task3

        # 确保两个协程都已经开始运行
        await both_started.wait()

        # 执行正常的调度流程
        await executor.execute()

        # 检查两个运行中任务都被取消
        self.assertEqual(cancelled, {"T2", "T3"})

        # 检查所有下游任务被阻塞
        self.assertEqual(t2.status, TaskStatus.BLOCKED)
        self.assertEqual(t3.status, TaskStatus.BLOCKED)
        self.assertEqual(t4.status, TaskStatus.BLOCKED)
        self.assertEqual(t5.status, TaskStatus.BLOCKED)

        # 检查协程全部退出
        self.assertTrue(task2.done())
        self.assertTrue(task3.done())

        # 检查运行任务映射清空
        self.assertEqual(executor.running_tasks, {})



if __name__ == "__main__":
    unittest.main()


