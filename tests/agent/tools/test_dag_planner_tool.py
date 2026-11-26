
import json
import sys
import types
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from nanobot.agent.tools.dag_planner import DAGPlannerTool


class TestDAGPlannerTool(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = MagicMock()
        self.tool = DAGPlannerTool(self.manager, 3)
        self.runtime = MagicMock()
        self.workflow_instance = MagicMock()
        self.workflow_instance.run = AsyncMock()
        self.workflow_class = MagicMock(return_value=self.workflow_instance)
        fake_workflow_module = types.ModuleType("nanobot.planner.workflow")
        fake_workflow_module.DAGWorkflow = self.workflow_class
        self.workflow_patcher = patch.dict(
            sys.modules,
            {"nanobot.planner.workflow": fake_workflow_module},
        )
        self.workflow_patcher.start()
        self.addCleanup(self.workflow_patcher.stop)

    async def execute_with_context(self, context, workflow_result, **kwargs):
        self.workflow_instance.run = AsyncMock(return_value=workflow_result)
        with patch(
            "nanobot.agent.tools.context.current_request_context",
            return_value=context,
        ):
            result = await self.tool.execute(**kwargs)
        return result, self.workflow_class, self.workflow_instance

    async def test_execute_passes_goal_and_normalized_disabled_tools(self):
        context = SimpleNamespace(
            runtime=self.runtime,
            original_user_text="请开发一个博客系统",
        )
        workflow_result = {
            "success": True,
            "terminal_results": [{"task_id": "T1"}],
            "process_results": [],
            "incomplete_tasks": [],
        }
        result, workflow_class, workflow_instance = (
            await self.execute_with_context(
                context,
                workflow_result,
                goal="开发博客系统",
                disabled_tools=[" web_search ", "exec", "web_search"],
            )
        )
        self.assertEqual(json.loads(result), workflow_result)
        workflow_class.assert_called_once_with(self.manager, 3)
        paras = workflow_instance.run.call_args.args[0]
        self.assertEqual(paras["goal"], "开发博客系统")
        self.assertEqual(
            paras["disabled_tools"],
            ["web_search", "exec"],
        )
        self.assertEqual(
            paras["original_user_text"],
            "请开发一个博客系统",
        )
        self.assertIs(
            workflow_instance.run.call_args.args[1],
            self.runtime,
        )

    async def test_execute_without_disabled_tools(self):
        context = SimpleNamespace(
            runtime=self.runtime,
            original_user_text="用户原始请求",
        )
        workflow_result = {
            "success": True,
            "terminal_results": [],
            "process_results": [],
            "incomplete_tasks": [],
        }
        _, _, workflow_instance = await self.execute_with_context(
            context,
            workflow_result,
            goal="执行一个任务",
        )
        paras = workflow_instance.run.call_args.args[0]
        self.assertNotIn("disabled_tools", paras)
        self.assertEqual(
            paras["original_user_text"],
            "用户原始请求",
        )

    async def test_execute_without_original_user_text(self):
        context = SimpleNamespace(
            runtime=self.runtime,
            original_user_text="   ",
        )
        workflow_result = {
            "success": True,
            "terminal_results": [],
            "process_results": [],
            "incomplete_tasks": [],
        }
        _, _, workflow_instance = await self.execute_with_context(
            context,
            workflow_result,
            goal="执行一个任务",
        )
        paras = workflow_instance.run.call_args.args[0]
        self.assertNotIn("original_user_text", paras)

    async def test_missing_request_context_raises_error(self):
        with patch(
            "nanobot.agent.tools.context.current_request_context",
            return_value=None,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "缺少当前请求的模型运行时",
            ):
                await self.tool.execute(goal="执行任务")

    async def test_missing_runtime_raises_error(self):
        context = SimpleNamespace(
            runtime=None,
            original_user_text="用户请求",
        )
        with patch(
            "nanobot.agent.tools.context.current_request_context",
            return_value=context,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "缺少当前请求的模型运行时",
            ):
                await self.tool.execute(goal="执行任务")

    async def test_disabled_tools_must_be_a_list(self):
        context = SimpleNamespace(
            runtime=self.runtime,
            original_user_text=None,
        )
        with patch(
            "nanobot.agent.tools.context.current_request_context",
            return_value=context,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "禁用工具列表格式有误",
            ):
                await self.tool.execute(
                    goal="执行任务",
                    disabled_tools="web_search",
                )

    async def test_disabled_tools_reject_invalid_names(self):
        context = SimpleNamespace(
            runtime=self.runtime,
            original_user_text=None,
        )
        invalid_values = [
            [""],
            ["   "],
            [None],
            [123],
        ]
        for disabled_tools in invalid_values:
            with self.subTest(disabled_tools=disabled_tools):
                with patch(
                    "nanobot.agent.tools.context.current_request_context",
                    return_value=context,
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "存在不合法工具名",
                    ):
                        await self.tool.execute(
                            goal="执行任务",
                            disabled_tools=disabled_tools,
                        )

    async def test_workflow_failure_is_preserved_in_tool_result(self):
        context = SimpleNamespace(
            runtime=self.runtime,
            original_user_text="开发博客系统",
        )
        workflow_result = {
            "success": False,
            "terminal_results": [{"task_id": "T1"}],
            "process_results": [],
            "incomplete_tasks": [
                {
                    "task_id": "T2",
                    "status": "failed",
                    "feedback": ["缺少必要文件"],
                }
            ],
        }
        result, _, _ = await self.execute_with_context(
            context,
            workflow_result,
            goal="开发博客系统",
        )
        parsed_result = json.loads(result)
        self.assertFalse(parsed_result["success"])
        self.assertEqual(
            parsed_result["incomplete_tasks"][0]["feedback"],
            ["缺少必要文件"],
        )

    async def test_workflow_exception_is_propagated(self):
        context = SimpleNamespace(
            runtime=self.runtime,
            original_user_text=None,
        )
        self.workflow_instance.run = AsyncMock(
            side_effect=RuntimeError("DAG 执行异常")
        )
        with (
            patch(
                "nanobot.agent.tools.context.current_request_context",
                return_value=context,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "DAG 执行异常",
            ),
        ):
            await self.tool.execute(goal="执行任务")

    async def test_tool_create_requires_subagent_manager(self):
        context = SimpleNamespace(
            subagent_manager=None,
            max_concurrent_planner_tasks=3,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "initialized subagent manager",
        ):
            DAGPlannerTool.create(context)

    async def test_tool_create_uses_context_configuration(self):
        context = SimpleNamespace(
            subagent_manager=self.manager,
            max_concurrent_planner_tasks=5,
        )
        tool = DAGPlannerTool.create(context)
        self.assertIsInstance(tool, DAGPlannerTool)
        self.assertIs(tool._manager, self.manager)
        self.assertEqual(tool.max_concurrent_planner_tasks, 5)