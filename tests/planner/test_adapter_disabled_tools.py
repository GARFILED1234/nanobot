
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from nanobot.planner.adapter import SubagentTaskExecutor
from nanobot.planner.graph import TaskGraph
from nanobot.planner.models import Task


class TestSubagentDisabledTools(unittest.IsolatedAsyncioTestCase):

    def make_request_context(self):
        return SimpleNamespace(
            channel="cli",
            chat_id="test-chat",
            session_key="cli:test-chat",
            original_user_text="完成一个测试任务",
            message_id="msg-001",
            runtime=object(),
            turn_id="turn-001",
            persist_session=True,
            log_content=True,
        )

    async def test_disabled_tools_are_forwarded_and_frozen(self):
        manager = SimpleNamespace(
            run_inline=AsyncMock(return_value="执行成功")
        )

        original_tools = ["web_search"]
        executor = SubagentTaskExecutor(
            manager,
            TaskGraph(),
            original_tools,
        )

        # 修改原列表，不应该影响 Executor 保存的集合
        original_tools.append("exec")

        self.assertEqual(
            executor.disabled_tools,
            frozenset({"web_search"}),
        )

        with (
            patch(
                "nanobot.planner.adapter.current_request_context",
                return_value=self.make_request_context(),
            ),
            patch(
                "nanobot.planner.adapter.current_workspace_scope",
                return_value=None,
            ),
        ):
            result = await executor(
                Task("T1", "执行测试任务")
            )

        self.assertEqual(result, "执行成功")

        manager.run_inline.assert_awaited_once()

        self.assertEqual(
            manager.run_inline.await_args.kwargs["disabled_tools"],
            frozenset({"web_search"}),
        )

    async def test_default_disabled_tools_is_empty(self):
        manager = SimpleNamespace(
            run_inline=AsyncMock(return_value="执行成功")
        )

        # 保持旧的两参数构造方式
        executor = SubagentTaskExecutor(
            manager,
            TaskGraph(),
        )

        with (
            patch(
                "nanobot.planner.adapter.current_request_context",
                return_value=self.make_request_context(),
            ),
            patch(
                "nanobot.planner.adapter.current_workspace_scope",
                return_value=None,
            ),
        ):
            await executor(Task("T1", "普通任务"))

        self.assertEqual(
            manager.run_inline.await_args.kwargs["disabled_tools"],
            frozenset(),
        )

    async def test_concurrent_executors_have_isolated_tool_limits(self):
        manager = SimpleNamespace(
            run_inline=AsyncMock(return_value="执行成功")
        )

        executor_a = SubagentTaskExecutor(
            manager,
            TaskGraph(),
            ["web_search"],
        )

        executor_b = SubagentTaskExecutor(
            manager,
            TaskGraph(),
            ["exec"],
        )

        with (
            patch(
                "nanobot.planner.adapter.current_request_context",
                return_value=self.make_request_context(),
            ),
            patch(
                "nanobot.planner.adapter.current_workspace_scope",
                return_value=None,
            ),
        ):
            await asyncio.gather(
                executor_a(Task("T1", "任务A")),
                executor_b(Task("T2", "任务B")),
            )

        self.assertEqual(
            manager.run_inline.await_count,
            2,
        )

        # 根据任务 ID 检查各自收到的限制
        received_limits = {
            call.kwargs["label"]: call.kwargs["disabled_tools"]
            for call in manager.run_inline.await_args_list
        }

        self.assertEqual(
            received_limits["T1"],
            frozenset({"web_search"}),
        )

        self.assertEqual(
            received_limits["T2"],
            frozenset({"exec"}),
        )


if __name__ == "__main__":
    unittest.main()
