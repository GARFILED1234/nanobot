
import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from nanobot.planner.planner import Planner


class TestPlanner(unittest.TestCase):

    def setUp(self):
        self.runtime = MagicMock()
        self.runtime.provider = MagicMock()
        self.runtime.provider.chat_with_retry = AsyncMock()
        self.runtime.model = "test-model"
        self.runtime.generation.max_tokens = 1000
        self.runtime.generation.temperature = 0.7
        self.runtime.generation.reasoning_effort = None

        self.planner = Planner(self.runtime)

    def _response(self, tasks):
        return MagicMock(
            content=json.dumps(
                {"tasks": tasks},
                ensure_ascii=False
            )
        )

    def _task(
        self,
        task_id="T1",
        description="实现博客后端",
        dependencies=None,
        acceptance_criteria=None,
    ):
        return {
            "task_id": task_id,
            "description": description,
            "dependencies": dependencies if dependencies is not None else [],
            "acceptance_criteria": (
                acceptance_criteria
                if acceptance_criteria is not None
                else ["实现指定功能", "通过对应测试"]
            ),
        }

    def test_plan_success(self):
        self.runtime.provider.chat_with_retry.return_value = self._response([
            self._task(
                "T1",
                "设计博客系统架构",
            ),
            self._task(
                "T2",
                "实现博客系统后端",
                ["T1"],
            ),
            self._task(
                "T3",
                "实现博客系统前端",
                ["T1"],
            ),
        ])

        import asyncio

        graph = asyncio.run(
            self.planner.plan({
                "goal": "开发一个博客系统"
            })
        )

        self.assertEqual(len(graph.tasks), 3)

        task1 = graph.get_task("T1")
        task2 = graph.get_task("T2")

        self.assertEqual(
            task1.acceptance_criteria,
            ["实现指定功能", "通过对应测试"],
        )
        self.assertEqual(task2.dependencies, ["T1"])

    def test_plan_invalid_goal(self):
        import asyncio

        invalid_inputs = [
            {},
            {"goal": ""},
            {"goal": "   "},
            {"goal": 123},
            {"goal": None},
        ]

        for paras in invalid_inputs:
            with self.subTest(paras=paras):
                with self.assertRaises(ValueError):
                    asyncio.run(self.planner.plan(paras))

    def test_plan_empty_response(self):
        import asyncio

        invalid_contents = [
            "",
            "   ",
            None,
            123,
        ]

        for content in invalid_contents:
            with self.subTest(content=content):
                self.runtime.provider.chat_with_retry.return_value = (
                    MagicMock(content=content)
                )

                with self.assertRaises(ValueError):
                    asyncio.run(
                        self.planner.plan({
                            "goal": "开发博客系统"
                        })
                    )

    def test_plan_invalid_json(self):
        import asyncio

        invalid_contents = [
            "not json",
            '{"tasks": "invalid"}',
            '{"tasks": []}',
            '[]',
            '{"other": []}',
        ]

        for content in invalid_contents:
            with self.subTest(content=content):
                self.runtime.provider.chat_with_retry.return_value = (
                    MagicMock(content=content)
                )

                with self.assertRaises(ValueError):
                    asyncio.run(
                        self.planner.plan({
                            "goal": "开发博客系统"
                        })
                    )

    def test_plan_circular_dependency(self):
        import asyncio

        self.runtime.provider.chat_with_retry.return_value = self._response([
            self._task("T1", dependencies=["T2"]),
            self._task("T2", dependencies=["T1"]),
        ])

        with self.assertRaises(ValueError):
            asyncio.run(
                self.planner.plan({
                    "goal": "开发博客系统"
                })
            )

    def test_plan_missing_dependency(self):
        import asyncio

        self.runtime.provider.chat_with_retry.return_value = self._response([
            self._task("T1", dependencies=["T99"]),
        ])

        with self.assertRaises(ValueError):
            asyncio.run(
                self.planner.plan({
                    "goal": "开发博客系统"
                })
            )

    def test_plan_parameters_validation(self):
        import asyncio

        self.runtime.provider.chat_with_retry.return_value = self._response([
            self._task(),
        ])

        valid_inputs = [
            {"goal": "开发博客系统"},
            {"goal": "开发博客系统", "parameters": None},
            {"goal": "开发博客系统", "parameters": {}},
        ]

        for paras in valid_inputs:
            with self.subTest(paras=paras):
                graph = asyncio.run(
                    self.planner.plan(paras)
                )
                self.assertEqual(len(graph.tasks), 1)

        invalid_inputs = [
            {"goal": "开发博客系统", "parameters": "invalid"},
            {"goal": "开发博客系统", "parameters": []},
            {"goal": "开发博客系统", "parameters": 123},
        ]

        for paras in invalid_inputs:
            with self.subTest(paras=paras):
                with self.assertRaises(ValueError):
                    asyncio.run(self.planner.plan(paras))

    def test_plan_llm_request_error(self):
        import asyncio

        self.runtime.provider.chat_with_retry.side_effect = RuntimeError(
            "LLM request failed"
        )

        with self.assertRaises(RuntimeError):
            asyncio.run(
                self.planner.plan({
                    "goal": "开发博客系统"
                })
            )

    def test_plan_forwards_original_user_text(self):
        import asyncio

        self.runtime.provider.chat_with_retry.return_value = self._response([
            self._task(
                description="使用 Python 和 SQLite 设计博客系统"
            )
        ])

        asyncio.run(
            self.planner.plan({
                "goal": "开发博客系统",
                "original_user_text": "请使用 Python 和 SQLite 开发博客系统",
            })
        )

        call_kwargs = (
            self.runtime.provider.chat_with_retry.call_args.kwargs
        )

        messages = call_kwargs["messages"]

        self.assertIn(
            "开发博客系统",
            messages[1]["content"],
        )
        self.assertIn(
            "original_user_text",
            messages[1]["content"],
        )


if __name__ == "__main__":
    unittest.main()
