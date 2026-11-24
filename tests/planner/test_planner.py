
import unittest
from unittest.mock import AsyncMock, MagicMock

from nanobot.planner.planner import Planner
from nanobot.planner.models import TaskStatus


class TestPlanner(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.runtime = MagicMock()
        self.runtime.model = "test-model"
        self.runtime.generation.max_tokens = 2048
        self.runtime.generation.temperature = 0.7
        self.runtime.generation.reasoning_effort = None

        self.runtime.provider.chat_with_retry = AsyncMock()

        self.planner = Planner(self.runtime)

    async def test_plan_success(self):
        """正常规划：LLM 返回 JSON，成功构建任务图。"""

        response = MagicMock()
        response.content = """
        {
            "tasks": [
                {
                    "task_id": "T1",
                    "description": "设计博客系统架构",
                    "dependencies": []
                },
                {
                    "task_id": "T2",
                    "description": "实现博客系统后端",
                    "dependencies": ["T1"]
                },
                {
                    "task_id": "T3",
                    "description": "实现博客系统前端",
                    "dependencies": ["T1"]
                }
            ]
        }
        """

        self.runtime.provider.chat_with_retry.return_value = response

        graph = await self.planner.plan({
            "goal": "开发一个博客系统"
        })

        self.assertEqual(len(graph.tasks), 3)

        self.assertEqual(
            graph.get_task("T1").description,
            "设计博客系统架构"
        )

        self.assertEqual(
            graph.get_task("T2").dependencies,
            ["T1"]
        )

        self.assertEqual(
            graph.get_task("T3").dependencies,
            ["T1"]
        )

        self.assertEqual(
            graph.get_task("T1").dependents,
            ["T2", "T3"]
        )

        self.runtime.provider.chat_with_retry.assert_awaited_once()

    async def test_plan_invalid_goal(self):
        """输入校验：拒绝非字符串和空目标。"""

        invalid_inputs = [
            {"goal": ""},
            {"goal": "   "},
            {"goal": 123},
            {"goal": None},
            {},
        ]

        for paras in invalid_inputs:
            with self.subTest(paras=paras):
                with self.assertRaises(ValueError):
                    await self.planner.plan(paras)

        self.runtime.provider.chat_with_retry.assert_not_awaited()

    async def test_plan_invalid_json(self):
        """JSON 校验：拒绝无法解析或结构不正确的内容。"""

        invalid_responses = [
            "not json",
            '{"tasks": "invalid"}',
            '{"tasks": []}',
            '[]',
            '{"other": []}',
        ]

        for content in invalid_responses:
            with self.subTest(content=content):
                response = MagicMock()
                response.content = content

                self.runtime.provider.chat_with_retry.return_value = response

                with self.assertRaises(ValueError):
                    await self.planner.plan({
                        "goal": "开发一个博客系统"
                    })

    async def test_plan_empty_response(self):
        """响应校验：拒绝空响应和非字符串响应内容。"""

        invalid_responses = [
            "",
            "   ",
            None,
            123,
        ]

        for content in invalid_responses:
            with self.subTest(content=content):
                response = MagicMock()
                response.content = content

                self.runtime.provider.chat_with_retry.return_value = response

                with self.assertRaises(ValueError):
                    await self.planner.plan({
                        "goal": "开发一个博客系统"
                    })

    async def test_plan_missing_dependency(self):
        """DAG 校验：拒绝依赖不存在的任务。"""

        response = MagicMock()
        response.content = """
        {
            "tasks": [
                {
                    "task_id": "T1",
                    "description": "实现博客后端",
                    "dependencies": ["T99"]
                }
            ]
        }
        """

        self.runtime.provider.chat_with_retry.return_value = response

        with self.assertRaises(ValueError):
            await self.planner.plan({
                "goal": "开发一个博客系统"
            })

    async def test_plan_circular_dependency(self):
        """DAG 校验：拒绝循环依赖。"""

        response = MagicMock()
        response.content = """
        {
            "tasks": [
                {
                    "task_id": "T1",
                    "description": "实现博客后端",
                    "dependencies": ["T2"]
                },
                {
                    "task_id": "T2",
                    "description": "实现博客前端",
                    "dependencies": ["T1"]
                }
            ]
        }
        """

        self.runtime.provider.chat_with_retry.return_value = response

        with self.assertRaises(ValueError):
            await self.planner.plan({
                "goal": "开发一个博客系统"
            })

    
    async def test_plan_parameters_validation(self):
        """输入校验：parameters 可选，但提供时必须是字典。"""

        # 准备合法的 LLM 响应
        response = MagicMock()
        response.content = """
        {
            "tasks": [
                {
                    "task_id": "T1",
                    "description": "开发博客系统",
                    "dependencies": []
                }
            ]
        }
        """
        self.runtime.provider.chat_with_retry.return_value = response

        # 第一组：允许的输入
        valid_inputs = [
            {"goal": "开发博客系统"},
            {"goal": "开发博客系统", "parameters": None},
            {"goal": "开发博客系统", "parameters": {}},
        ]

        for paras in valid_inputs:
            with self.subTest(paras=paras):
                graph = await self.planner.plan(paras)
                self.assertEqual(len(graph.tasks), 1)

        # 第二组：拒绝的输入
        invalid_inputs = [
            {"goal": "开发博客系统", "parameters": "错误参数"},
            {"goal": "开发博客系统", "parameters": []},
            {"goal": "开发博客系统", "parameters": 123},
        ]

        for paras in invalid_inputs:
            with self.subTest(paras=paras):
                with self.assertRaises(ValueError):
                    await self.planner.plan(paras)

        
    async def test_plan_llm_request_error(self):
        """LLM 请求失败：异常应向上传播，不应返回空任务图。"""

        self.runtime.provider.chat_with_retry.side_effect = RuntimeError(
            "LLM API 请求失败"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "LLM API 请求失败"
        ):
            await self.planner.plan({
                "goal": "开发一个博客系统"
            })

        self.runtime.provider.chat_with_retry.assert_awaited_once()




if __name__ == "__main__":
    unittest.main()
