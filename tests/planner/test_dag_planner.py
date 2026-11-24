
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from nanobot.agent.tools.dag_planner import DAGPlannerTool


class TestDAGPlannerTool(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.manager = object()
        self.max_concurrency = 3

        self.tool = DAGPlannerTool(
            self.manager,
            self.max_concurrency,
        )

        self.runtime = object()

        self.request_context = SimpleNamespace(
            runtime=self.runtime,
        )

    async def test_execute_success(self):
        """正常执行：正确调用 Workflow，并返回合法 JSON 字符串。"""

        workflow_result = {
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
        }

        with (
            patch(
                "nanobot.agent.tools.context.current_request_context",
                return_value=self.request_context,
            ),
            patch(
                "nanobot.planner.workflow.DAGWorkflow",
            ) as MockWorkflow,
        ):
            MockWorkflow.return_value.run = AsyncMock(
                return_value=workflow_result,
            )

            result = await self.tool.execute(
                goal="开发一个博客系统",
            )

        self.assertIsInstance(result, str)

        result_dict = json.loads(result)

        self.assertEqual(result_dict, workflow_result)

        MockWorkflow.assert_called_once_with(
            self.manager,
            self.max_concurrency,
        )

        MockWorkflow.return_value.run.assert_awaited_once_with(
            {"goal": "开发一个博客系统"},
            self.runtime,
        )

    async def test_execute_workflow_error(self):
        """Workflow 执行失败：异常向上传播。"""

        with (
            patch(
                "nanobot.agent.tools.context.current_request_context",
                return_value=self.request_context,
            ),
            patch(
                "nanobot.planner.workflow.DAGWorkflow",
            ) as MockWorkflow,
        ):
            MockWorkflow.return_value.run = AsyncMock(
                side_effect=RuntimeError("DAG 工作流执行失败"),
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "DAG 工作流执行失败",
            ):
                await self.tool.execute(
                    goal="开发一个博客系统",
                )

            MockWorkflow.assert_called_once_with(
                self.manager,
                self.max_concurrency,
            )

            MockWorkflow.return_value.run.assert_awaited_once_with(
                {"goal": "开发一个博客系统"},
                self.runtime,
            )

    async def test_execute_missing_context(self):
        """缺少请求上下文时，拒绝执行。"""

        with patch(
            "nanobot.agent.tools.context.current_request_context",
            return_value=None,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "缺少当前请求的模型运行时",
            ):
                await self.tool.execute(
                    goal="开发博客系统",
                )

    async def test_execute_missing_runtime(self):
        """请求上下文存在，但 runtime 缺失。"""

        context = SimpleNamespace(runtime=None)

        with patch(
            "nanobot.agent.tools.context.current_request_context",
            return_value=context,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "缺少当前请求的模型运行时",
            ):
                await self.tool.execute(
                    goal="开发博客系统",
                )

    async def test_execute_parameters_forwarding(self):
        """验证所有额外参数完整传递给 Workflow。"""

        parameters = {
            "task1": "设计数据库",
            "task2": "开发后端",
            "dependencies": {
                "T2": ["T1"],
            },
        }

        kwargs = {
            "goal": "开发一个博客系统",
            "parameters": parameters,
        }

        with (
            patch(
                "nanobot.agent.tools.context.current_request_context",
                return_value=self.request_context,
            ),
            patch(
                "nanobot.planner.workflow.DAGWorkflow",
            ) as MockWorkflow,
        ):
            MockWorkflow.return_value.run = AsyncMock(
                return_value={
                    "terminal_results": [],
                    "process_results": [],
                    "incomplete_tasks": [],
                },
            )

            await self.tool.execute(**kwargs)

            MockWorkflow.return_value.run.assert_awaited_once_with(
                kwargs,
                self.runtime,
            )

    async def test_create_success(self):
        """create 能正确从 ToolContext 注入依赖。"""

        context = SimpleNamespace(
            subagent_manager=self.manager,
            max_concurrent_planner_tasks=self.max_concurrency,
        )

        tool = DAGPlannerTool.create(context)

        self.assertIsInstance(tool, DAGPlannerTool)

        self.assertIs(
            tool._manager,
            self.manager,
        )

        self.assertEqual(
            tool.max_concurrent_planner_tasks,
            self.max_concurrency,
        )

    async def test_create_without_subagent_manager(self):
        """没有初始化 SubagentManager 时拒绝创建工具。"""

        context = SimpleNamespace(
            subagent_manager=None,
            max_concurrent_planner_tasks=self.max_concurrency,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "initialized subagent manager",
        ):
            DAGPlannerTool.create(context)


if __name__ == "__main__":
    unittest.main()
