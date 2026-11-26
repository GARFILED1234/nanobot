
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
        self.original_user_text = "使用 Python 和 SQLite 开发博客系统"

        self.request_context = SimpleNamespace(
            runtime=self.runtime,
            original_user_text=self.original_user_text,
        )

    async def test_execute_success(self):
        """原始请求存在时，正确传递给 Workflow。"""

        workflow_result = {
            "terminal_results": [],
            "process_results": [],
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
                goal="开发博客系统",
            )

        self.assertEqual(
            json.loads(result),
            workflow_result,
        )

        MockWorkflow.return_value.run.assert_awaited_once_with(
            {
                "goal": "开发博客系统",
                "original_user_text": self.original_user_text,
            },
            self.runtime,
        )

    async def test_execute_without_original_user_text(self):
        """原始请求不存在时，仍然可以正常执行。"""

        context = SimpleNamespace(
            runtime=self.runtime,
            original_user_text=None,
        )

        with (
            patch(
                "nanobot.agent.tools.context.current_request_context",
                return_value=context,
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

            await self.tool.execute(
                goal="开发博客系统",
            )

        MockWorkflow.return_value.run.assert_awaited_once_with(
            {
                "goal": "开发博客系统",
            },
            self.runtime,
        )

    async def test_execute_with_empty_original_user_text(self):
        """原始请求为空字符串时，不传递该字段。"""

        context = SimpleNamespace(
            runtime=self.runtime,
            original_user_text="   ",
        )

        with (
            patch(
                "nanobot.agent.tools.context.current_request_context",
                return_value=context,
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

            await self.tool.execute(
                goal="开发博客系统",
            )

        MockWorkflow.return_value.run.assert_awaited_once_with(
            {
                "goal": "开发博客系统",
            },
            self.runtime,
        )

    async def test_execute_workflow_error(self):
        """Workflow 执行失败时，异常向上传播。"""

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
                    goal="开发博客系统",
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

        context = SimpleNamespace(
            runtime=None,
            original_user_text=self.original_user_text,
        )

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
        """额外参数和原始请求都能正确传递，且不修改原始 kwargs。"""

        parameters = {
            "task1": "设计数据库",
            "task2": "开发后端",
            "dependencies": {
                "T2": ["T1"],
            },
        }

        kwargs = {
            "goal": "开发博客系统",
            "parameters": parameters,
        }

        original_kwargs = kwargs.copy()

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

        expected = {
            "goal": "开发博客系统",
            "parameters": parameters,
            "original_user_text": self.original_user_text,
        }

        MockWorkflow.return_value.run.assert_awaited_once_with(
            expected,
            self.runtime,
        )

        self.assertEqual(kwargs, original_kwargs)

    async def test_create_success(self):
        """create 能正确从 ToolContext 注入依赖。"""

        context = SimpleNamespace(
            subagent_manager=self.manager,
            max_concurrent_planner_tasks=self.max_concurrency,
        )

        tool = DAGPlannerTool.create(context)

        self.assertIsInstance(tool, DAGPlannerTool)
        self.assertIs(tool._manager, self.manager)

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
