
import unittest
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from nanobot.agent.tools.base import ToolResult
from nanobot.session.manager import SessionPolicy

from nanobot.planner.adapter import SubagentTaskExecutor
from nanobot.planner.models import Task
from nanobot.planner.graph import TaskGraph


class TestSubagentTaskExecutor(unittest.IsolatedAsyncioTestCase):

    async def test_execute_success(self):
        manager = SimpleNamespace(
            run_inline=AsyncMock(return_value="任务执行成功")
        )

        runtime = object()
        workspace_scope = object()

        request_ctx = SimpleNamespace(
            channel="cli",
            chat_id="test-chat",
            session_key="cli:test-chat",
            message_id="msg-001",
            runtime=runtime,
            turn_id="turn-001",
            persist_session=True,
            log_content=True,
            original_user_text="测试用户原始需求",
        )

        task = Task(
            task_id="T1",
            description="完成一个测试任务",
        )

        graph = TaskGraph()
        executor = SubagentTaskExecutor(manager, graph)

        with (
            patch(
                "nanobot.planner.adapter.current_request_context",
                return_value=request_ctx,
            ),
            patch(
                "nanobot.planner.adapter.current_workspace_scope",
                return_value=workspace_scope,
            ),
        ):
            result = await executor(task)

        self.assertEqual(result, "任务执行成功")

        manager.run_inline.assert_awaited_once()

        prompt = manager.run_inline.await_args.kwargs["task"]

        self.assertIn("测试用户原始需求", prompt)
        self.assertNotIn(
            "前面列出的前置任务均已完成",
            prompt,
        )
        self.assertNotIn("前置任务：", prompt)
        self.assertIn("当前任务：完成一个测试任务", prompt)

        self.assertEqual(
            manager.run_inline.await_args.kwargs,
            {
                "task": prompt,
                "label": "T1",
                "origin_channel": "cli",
                "origin_chat_id": "test-chat",
                "session_key": "cli:test-chat",
                "origin_message_id": "msg-001",
                "runtime": runtime,
                "origin_turn_id": "turn-001",
                "workspace_scope": workspace_scope,
                "session_policy": SessionPolicy(
                    persist=True,
                    log_content=True,
                ),
                "disabled_tools": frozenset(),
            },
        )

    async def test_execute_with_dependencies(self):
        manager = SimpleNamespace(
            run_inline=AsyncMock(return_value="分析完成")
        )

        request_ctx = SimpleNamespace(
            channel="cli",
            chat_id="test-chat",
            session_key="cli:test-chat",
            message_id="msg-001",
            runtime=object(),
            turn_id="turn-001",
            persist_session=True,
            log_content=True,
            original_user_text="测试用户原始需求",
        )

        graph = TaskGraph()

        pre_task = Task(
            task_id="T1",
            description="搜索资料",
        )
        pre_task.result = "搜索到的资料内容"

        current_task = Task(
            task_id="T2",
            description="分析资料并总结核心观点",
        )

        graph.add_task(pre_task)
        graph.add_task(current_task)
        graph.add_dependency("T2", "T1")

        executor = SubagentTaskExecutor(manager, graph)

        with (
            patch(
                "nanobot.planner.adapter.current_request_context",
                return_value=request_ctx,
            ),
            patch(
                "nanobot.planner.adapter.current_workspace_scope",
                return_value=None,
            ),
        ):
            result = await executor(current_task)

        self.assertEqual(result, "分析完成")

        manager.run_inline.assert_awaited_once()

        prompt = manager.run_inline.await_args.kwargs["task"]

        self.assertIn("T1", prompt)
        self.assertIn("搜索资料", prompt)
        self.assertIn("搜索到的资料内容", prompt)
        self.assertIn("前面列出的前置任务均已完成", prompt)
        self.assertIn(
            "当前任务：分析资料并总结核心观点",
            prompt,
        )

    async def test_execute_with_multiple_dependencies(self):
        manager = SimpleNamespace(
            run_inline=AsyncMock(return_value="综合分析完成")
        )

        request_ctx = SimpleNamespace(
            channel="cli",
            chat_id="test-chat",
            session_key="cli:test-chat",
            message_id="msg-001",
            runtime=object(),
            turn_id="turn-001",
            persist_session=True,
            log_content=True,
            original_user_text="测试用户原始需求",
        )

        graph = TaskGraph()

        task1 = Task("T1", "搜索技术资料")
        task1.result = "技术资料内容"

        task2 = Task("T2", "搜索行业案例")
        task2.result = "行业案例内容"

        task3 = Task("T3", "综合分析技术资料和行业案例")

        graph.add_task(task1)
        graph.add_task(task2)
        graph.add_task(task3)

        graph.add_dependency("T3", "T1")
        graph.add_dependency("T3", "T2")

        executor = SubagentTaskExecutor(manager, graph)

        with (
            patch(
                "nanobot.planner.adapter.current_request_context",
                return_value=request_ctx,
            ),
            patch(
                "nanobot.planner.adapter.current_workspace_scope",
                return_value=None,
            ),
        ):
            result = await executor(task3)

        self.assertEqual(result, "综合分析完成")

        manager.run_inline.assert_awaited_once()

        prompt = manager.run_inline.await_args.kwargs["task"]

        self.assertIn("T1", prompt)
        self.assertIn("技术资料内容", prompt)
        self.assertIn("T2", prompt)
        self.assertIn("行业案例内容", prompt)
        self.assertIn(
            "综合分析技术资料和行业案例",
            prompt,
        )

    async def test_execute_error(self):
        manager = SimpleNamespace(
            run_inline=AsyncMock(
                return_value=ToolResult.error("子任务执行失败")
            )
        )

        request_ctx = SimpleNamespace(
            channel="cli",
            chat_id="test-chat",
            session_key="cli:test-chat",
            message_id="msg-001",
            runtime=object(),
            turn_id="turn-001",
            persist_session=True,
            log_content=True,
            original_user_text="测试用户原始需求",
        )

        task = Task(
            task_id="T1",
            description="执行失败的任务",
        )

        graph = TaskGraph()
        executor = SubagentTaskExecutor(manager, graph)

        with (
            patch(
                "nanobot.planner.adapter.current_request_context",
                return_value=request_ctx,
            ),
            patch(
                "nanobot.planner.adapter.current_workspace_scope",
                return_value=None,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "子任务执行失败",
            ):
                await executor(task)

    async def test_execute_cancelled(self):
        manager = SimpleNamespace(
            run_inline=AsyncMock(
                side_effect=asyncio.CancelledError
            )
        )

        request_ctx = SimpleNamespace(
            channel="cli",
            chat_id="test-chat",
            session_key="cli:test-chat",
            message_id="msg-001",
            runtime=object(),
            turn_id="turn-001",
            persist_session=True,
            log_content=True,
            original_user_text="测试用户原始需求",
        )

        task = Task(
            task_id="T1",
            description="执行中的任务",
        )

        graph = TaskGraph()
        executor = SubagentTaskExecutor(manager, graph)

        with (
            patch(
                "nanobot.planner.adapter.current_request_context",
                return_value=request_ctx,
            ),
            patch(
                "nanobot.planner.adapter.current_workspace_scope",
                return_value=None,
            ),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await executor(task)

    async def test_execute_without_request_context(self):
        manager = SimpleNamespace(
            run_inline=AsyncMock()
        )

        task = Task(
            task_id="T1",
            description="测试上下文缺失",
        )

        graph = TaskGraph()
        executor = SubagentTaskExecutor(manager, graph)

        with patch(
            "nanobot.planner.adapter.current_request_context",
            return_value=None,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "缺少当前请求的模型运行时",
            ):
                await executor(task)

        manager.run_inline.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
