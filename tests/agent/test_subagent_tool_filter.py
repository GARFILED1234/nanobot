
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from nanobot.agent.memory import Consolidator
from nanobot.agent.subagent import SubagentManager
from nanobot.bus.queue import MessageBus


class TestSubagentToolFilter(unittest.TestCase):

    def setUp(self):
        self.workspace = Path("tests/tmp_subagent_tool_filter")
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.manager = SubagentManager(
            workspace=self.workspace,
            bus=MessageBus(),
            max_tool_result_chars=16_000,
            consolidator=MagicMock(spec=Consolidator),
        )

    def test_build_tools_registers_web_search_by_default(self):
        tools = self.manager._build_tools()

        self.assertTrue(tools.has("web_search"))

    def test_build_tools_unregisters_specified_tool(self):
        tools = self.manager._build_tools(
            disabled_tools={"web_search"}
        )

        self.assertFalse(tools.has("web_search"))

    def test_build_tools_only_unregisters_specified_tools(self):
        tools = self.manager._build_tools(
            disabled_tools={"web_search"}
        )

        self.assertTrue(tools.has("web_fetch"))
        self.assertTrue(tools.has("read_file"))

    def test_build_tools_filter_does_not_affect_next_registry(self):
        restricted_tools = self.manager._build_tools(
            disabled_tools={"web_search"}
        )
        normal_tools = self.manager._build_tools()

        self.assertFalse(restricted_tools.has("web_search"))
        self.assertTrue(normal_tools.has("web_search"))

    def test_build_tools_filter_does_not_modify_other_registry(self):
        normal_tools = self.manager._build_tools()
        restricted_tools = self.manager._build_tools(
            disabled_tools={"web_search"}
        )

        self.assertTrue(normal_tools.has("web_search"))
        self.assertFalse(restricted_tools.has("web_search"))


if __name__ == "__main__":
    unittest.main()
