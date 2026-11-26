
import tempfile
import unittest
from pathlib import Path

from nanobot.planner.models import AcceptanceCriterion
from nanobot.planner.validation.registry import create_default_registry
from nanobot.planner.validation.validator import TaskResultValidator


class TestValidatorPathException(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workspace_root = Path(self.temp_dir.name)
        self.validator = TaskResultValidator(create_default_registry())

    def _criterion(self, criterion_id, criterion_type, parameters):
        return AcceptanceCriterion(
            criterion_id=criterion_id,
            type=criterion_type,
            description="测试验收标准",
            parameters=parameters,
        )

    def test_path_exception_does_not_interrupt_remaining_criteria(self):
        target = self.workspace_root / "app.py"
        target.write_text("print('hello')", encoding="utf-8")

        criteria = [
            self._criterion(
                "C1",
                "file_exists",
                {"path": " app.py "},
            ),
            self._criterion(
                "C2",
                "file_exists",
                {"path": "app.py"},
            ),
        ]

        result = self.validator.validate(
            criteria,
            self.workspace_root,
        )

        self.assertFalse(result.passed)
        self.assertEqual(len(result.criterion_results), 2)

        first_result = result.criterion_results[0]
        second_result = result.criterion_results[1]

        self.assertEqual(first_result.criterion_id, "C1")
        self.assertFalse(first_result.passed)
        self.assertTrue(
            any("文件路径不能包含首尾空格" in item for item in first_result.feedback)
        )

        self.assertEqual(second_result.criterion_id, "C2")
        self.assertTrue(second_result.passed)

    def test_file_contains_path_exception_is_caught(self):
        target = self.workspace_root / "app.py"
        target.write_text("hello nanobot", encoding="utf-8")

        criterion = self._criterion(
            "C1",
            "file_contains",
            {"path": " app.py ", "text": "nanobot"},
        )

        result = self.validator.validate(
            [criterion],
            self.workspace_root,
        )

        self.assertFalse(result.passed)
        self.assertEqual(len(result.criterion_results), 1)
        self.assertFalse(result.criterion_results[0].passed)
        self.assertTrue(
            any("文件路径不能包含首尾空格" in item for item in result.feedback)
        )


if __name__ == "__main__":
    unittest.main()
