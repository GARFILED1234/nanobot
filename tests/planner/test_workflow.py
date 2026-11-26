
import unittest
from unittest.mock import AsyncMock, patch

from nanobot.planner.workflow import DAGWorkflow
from nanobot.planner.graph import TaskGraph
from nanobot.planner.models import Task, TaskStatus


class TestDAGWorkflow(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.manager = object()
        self.runtime = object()
        self.max_concurrency = 3

        self.workflow = DAGWorkflow(
            self.manager,
            self.max_concurrency,
        )

        self.paras = {
            "goal": "开发一个博客系统",
        }

    def _build_graph(self):
        """构造包含成功、失败、阻塞和取消任务的 DAG。"""

        graph = TaskGraph()

        tasks = [
            Task("T1", "设计数据库"),
            Task("T2", "开发后端接口"),
            Task("T3", "开发前端页面"),
            Task("T4", "编写测试"),
            Task("T5", "部署项目"),
        ]

        for task in tasks:
            graph.add_task(task)

        graph.add_dependency("T2", "T1")

        return graph

    async def test_run_success(self):
        """正常执行：验证 Planner、Adapter 和 DAGExecutor 的协作。"""

        graph = TaskGraph()

        graph.add_task(Task("T1", "设计数据库"))
        graph.add_task(Task("T2", "开发后端接口"))
        graph.add_dependency("T2", "T1")

        async def fake_execute():
            t1 = graph.get_task("T1")
            t1.status = TaskStatus.COMPLETED
            t1.result = "数据库设计完成"

            t2 = graph.get_task("T2")
            t2.status = TaskStatus.COMPLETED
            t2.result = "后端接口开发完成"

        with (
            patch(
                "nanobot.planner.workflow.Planner",
            ) as MockPlanner,
            patch(
                "nanobot.planner.workflow.SubagentTaskExecutor",
            ) as MockSubagentExecutor,
            patch(
                "nanobot.planner.workflow.DAGExecutor",
            ) as MockDAGExecutor,
        ):
            MockPlanner.return_value.plan = AsyncMock(
                return_value=graph,
            )

            MockDAGExecutor.return_value.execute = AsyncMock(
                side_effect=fake_execute,
            )

            result = await self.workflow.run(
                self.paras,
                self.runtime,
            )

        self.assertEqual(
            result,
            {
                "terminal_results": [
                    {
                        "task_id": "T2",
                        "description": "开发后端接口",
                        "status": "completed",
                        "result": "后端接口开发完成",
                    }
                ],
                "process_results": [
                    {
                        "task_id": "T1",
                        "description": "设计数据库",
                        "status": "completed",
                        "result": "数据库设计完成",
                    }
                ],
                "incomplete_tasks": [],
            },
        )

        MockPlanner.assert_called_once_with(self.runtime)

        MockPlanner.return_value.plan.assert_awaited_once_with(
            self.paras,
        )

        MockSubagentExecutor.assert_called_once_with(
            self.manager,
            graph,
            None,
        )

        MockDAGExecutor.assert_called_once_with(
            graph,
            MockSubagentExecutor.return_value,
            self.max_concurrency,
        )

        MockDAGExecutor.return_value.execute.assert_awaited_once()

    async def test_run_planner_error(self):
        """Planner 规划失败：不应该创建执行组件。"""

        with (
            patch(
                "nanobot.planner.workflow.Planner",
            ) as MockPlanner,
            patch(
                "nanobot.planner.workflow.SubagentTaskExecutor",
            ) as MockSubagentExecutor,
            patch(
                "nanobot.planner.workflow.DAGExecutor",
            ) as MockDAGExecutor,
        ):
            MockPlanner.return_value.plan = AsyncMock(
                side_effect=RuntimeError("Planner 规划失败"),
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "Planner 规划失败",
            ):
                await self.workflow.run(
                    self.paras,
                    self.runtime,
                )

            MockPlanner.assert_called_once_with(self.runtime)

            MockPlanner.return_value.plan.assert_awaited_once_with(
                self.paras,
            )

            MockSubagentExecutor.assert_not_called()
            MockDAGExecutor.assert_not_called()

    async def test_run_executor_error(self):
        """DAGExecutor 调度失败：异常应该向上传播。"""

        graph = TaskGraph()
        graph.add_task(Task("T1", "设计数据库"))

        with (
            patch(
                "nanobot.planner.workflow.Planner",
            ) as MockPlanner,
            patch(
                "nanobot.planner.workflow.SubagentTaskExecutor",
            ) as MockSubagentExecutor,
            patch(
                "nanobot.planner.workflow.DAGExecutor",
            ) as MockDAGExecutor,
        ):
            MockPlanner.return_value.plan = AsyncMock(
                return_value=graph,
            )

            MockDAGExecutor.return_value.execute = AsyncMock(
                side_effect=RuntimeError("DAG 调度失败"),
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "DAG 调度失败",
            ):
                await self.workflow.run(
                    self.paras,
                    self.runtime,
                )

            MockPlanner.return_value.plan.assert_awaited_once_with(
                self.paras,
            )

            MockSubagentExecutor.assert_called_once_with(
                self.manager,
                graph,
                None,
            )

            MockDAGExecutor.assert_called_once_with(
                graph,
                MockSubagentExecutor.return_value,
                self.max_concurrency,
            )

            MockDAGExecutor.return_value.execute.assert_awaited_once()

    async def test_run_result_classification(self):
        """验证成功、失败、阻塞和取消任务的结果分类。"""

        graph = self._build_graph()

        async def fake_execute():
            t1 = graph.get_task("T1")
            t1.status = TaskStatus.COMPLETED
            t1.result = "数据库设计完成"

            t2 = graph.get_task("T2")
            t2.status = TaskStatus.COMPLETED
            t2.result = "后端接口开发完成"

            t3 = graph.get_task("T3")
            t3.status = TaskStatus.FAILED
            t3.error = "前端开发失败"

            t4 = graph.get_task("T4")
            t4.status = TaskStatus.BLOCKED

            t5 = graph.get_task("T5")
            t5.status = TaskStatus.CANCELLED

        with (
            patch(
                "nanobot.planner.workflow.Planner",
            ) as MockPlanner,
            patch(
                "nanobot.planner.workflow.SubagentTaskExecutor",
            ),
            patch(
                "nanobot.planner.workflow.DAGExecutor",
            ) as MockDAGExecutor,
        ):
            MockPlanner.return_value.plan = AsyncMock(
                return_value=graph,
            )

            MockDAGExecutor.return_value.execute = AsyncMock(
                side_effect=fake_execute,
            )

            result = await self.workflow.run(
                self.paras,
                self.runtime,
            )

        terminal_results = result["terminal_results"]
        process_results = result["process_results"]
        incomplete_tasks = result["incomplete_tasks"]

        self.assertEqual(
            [task["task_id"] for task in terminal_results],
            ["T2"],
        )

        self.assertEqual(
            [task["task_id"] for task in process_results],
            ["T1"],
        )

        self.assertEqual(
            {task["task_id"] for task in incomplete_tasks},
            {"T3", "T4", "T5"},
        )

        incomplete_by_id = {
            task["task_id"]: task
            for task in incomplete_tasks
        }

        self.assertEqual(
            incomplete_by_id["T3"]["error"],
            "前端开发失败",
        )

        self.assertEqual(
            incomplete_by_id["T4"]["fail_reason"],
            "因前置依赖失败而未执行",
        )

        self.assertEqual(
            incomplete_by_id["T5"]["fail_reason"],
            "任务已取消",
        )

        self.assertNotIn(
            "result",
            incomplete_by_id["T3"],
        )

        all_task_ids = (
            [task["task_id"] for task in terminal_results]
            + [task["task_id"] for task in process_results]
            + [task["task_id"] for task in incomplete_tasks]
        )

        self.assertEqual(len(all_task_ids), 5)
        self.assertEqual(len(set(all_task_ids)), 5)


if __name__ == "__main__":
    unittest.main()
