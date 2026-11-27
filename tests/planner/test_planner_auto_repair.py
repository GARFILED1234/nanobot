
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from nanobot.planner.planner import Planner


class TestPlannerAutoRepair(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.runtime = MagicMock()
        self.runtime.provider.chat_with_retry = AsyncMock()
        self.runtime.model = "test-model"
        self.runtime.generation.max_tokens = 100
        self.runtime.generation.temperature = 0
        self.runtime.generation.reasoning_effort = None
        self.planner = Planner(self.runtime)
        self.planner._build_system_prompt = MagicMock(return_value="system prompt")
        self.planner._build_task_graph = MagicMock(return_value="test graph")

    def _valid_response(self):
        return json.dumps({
            "tasks": [
                {
                    "task_id": "T1",
                    "description": "创建应用文件",
                    "dependencies": [],
                    "acceptance_criteria": [
                        {
                            "criterion_id": "C1",
                            "type": "file_exists",
                            "description": "检查文件是否存在",
                            "parameters": {"path": "app.py"}
                        }
                    ]
                }
            ]
        })

    def _invalid_response(self):
        return json.dumps({
            "tasks": [
                {
                    "task_id": "T1",
                    "description": "创建应用文件",
                    "dependencies": [],
                    "acceptance_criteria": [
                        {
                            "criterion_id": "C1",
                            "type": "file_contains",
                            "description": "检查文件内容",
                            "parameters": {"path": "app.py"}
                        }
                    ]
                }
            ]
        })

    async def test_first_parse_success_without_repair(self):
        self.runtime.provider.chat_with_retry.return_value = SimpleNamespace(
            content=self._valid_response()
        )
        result = await self.planner.plan({"goal": "创建应用"})
        self.assertEqual(result, "test graph")
        self.assertEqual(self.runtime.provider.chat_with_retry.await_count, 1)
        self.planner._build_task_graph.assert_called_once()

    async def test_first_parse_failure_repair_success(self):
        self.runtime.provider.chat_with_retry.side_effect = [
            SimpleNamespace(content=self._invalid_response()),
            SimpleNamespace(content=self._valid_response())
        ]
        result = await self.planner.plan({"goal": "创建应用"})
        self.assertEqual(result, "test graph")
        self.assertEqual(self.runtime.provider.chat_with_retry.await_count, 2)
        self.planner._build_task_graph.assert_called_once()
        repair_messages = self.runtime.provider.chat_with_retry.await_args_list[1].kwargs["messages"]
        self.assertEqual(repair_messages[-2]["content"], self._invalid_response())
        self.assertIn("缺少参数", repair_messages[-1]["content"])

    async def test_repair_failure_does_not_build_task_graph(self):
        self.runtime.provider.chat_with_retry.side_effect = [
            SimpleNamespace(content=self._invalid_response()),
            SimpleNamespace(content=self._invalid_response())
        ]
        with self.assertRaisesRegex(ValueError, "Planner自动修复失败"):
            await self.planner.plan({"goal": "创建应用"})
        self.assertEqual(self.runtime.provider.chat_with_retry.await_count, 2)
        self.planner._build_task_graph.assert_not_called()

    async def test_first_llm_request_exception_does_not_trigger_repair(self):
        self.runtime.provider.chat_with_retry.side_effect = RuntimeError("network error")
        with self.assertRaisesRegex(RuntimeError, "network error"):
            await self.planner.plan({"goal": "创建应用"})
        self.assertEqual(self.runtime.provider.chat_with_retry.await_count, 1)
        self.planner._build_task_graph.assert_not_called()


if __name__ == "__main__":
    unittest.main()
