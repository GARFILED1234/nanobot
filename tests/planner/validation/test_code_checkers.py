import tempfile
import unittest
from pathlib import Path

from nanobot.planner.models import AcceptanceCriterion
from nanobot.planner.validation.checkers.code import (
    PythonSyntaxChecker,
    JsonValidChecker,
)


class TestPythonSyntaxChecker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp_dir.name)
        self.checker = PythonSyntaxChecker()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_criterion(self, path):
        return AcceptanceCriterion(
            criterion_id="C1",
            type="python_syntax",
            description="检查 Python 语法",
            parameters={"path": path},
        )

    def test_valid_python_file(self):
        (self.workspace_root / "test.py").write_text(
            "def hello():\n    return 'hello'\n",
            encoding="utf-8",
        )
        criterion = self._create_criterion("test.py")
        result = self.checker.validate(criterion, self.workspace_root)
        self.assertTrue(result.passed)
        self.assertEqual(result.criterion_id, "C1")

    def test_invalid_python_file(self):
        (self.workspace_root / "test.py").write_text(
            "def hello(\n",
            encoding="utf-8",
        )
        criterion = self._create_criterion("test.py")
        result = self.checker.validate(criterion, self.workspace_root)
        self.assertFalse(result.passed)
        self.assertTrue(any("Python 语法错误" in item for item in result.feedback))

    def test_file_not_exist(self):
        criterion = self._create_criterion("missing.py")
        result = self.checker.validate(criterion, self.workspace_root)
        self.assertFalse(result.passed)

    def test_path_outside_workspace(self):
        criterion = self._create_criterion("../outside.py")
        result = self.checker.validate(criterion, self.workspace_root)
        self.assertFalse(result.passed)


class TestJsonValidChecker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp_dir.name)
        self.checker = JsonValidChecker()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_criterion(self, path):
        return AcceptanceCriterion(
            criterion_id="C2",
            type="json_valid",
            description="检查 JSON 格式",
            parameters={"path": path},
        )

    def test_valid_json_object(self):
        (self.workspace_root / "test.json").write_text(
            '{"name": "nanobot", "version": 1}',
            encoding="utf-8",
        )
        criterion = self._create_criterion("test.json")
        result = self.checker.validate(criterion, self.workspace_root)
        self.assertTrue(result.passed)

    def test_valid_json_array(self):
        (self.workspace_root / "test.json").write_text(
            '[1, 2, 3]',
            encoding="utf-8",
        )
        criterion = self._create_criterion("test.json")
        result = self.checker.validate(criterion, self.workspace_root)
        self.assertTrue(result.passed)

    def test_valid_json_scalar(self):
        (self.workspace_root / "test.json").write_text(
            '"hello"',
            encoding="utf-8",
        )
        criterion = self._create_criterion("test.json")
        result = self.checker.validate(criterion, self.workspace_root)
        self.assertTrue(result.passed)

    def test_invalid_json(self):
        (self.workspace_root / "test.json").write_text(
            '{"name": "nanobot"',
            encoding="utf-8",
        )
        criterion = self._create_criterion("test.json")
        result = self.checker.validate(criterion, self.workspace_root)
        self.assertFalse(result.passed)
        self.assertTrue(any("json 语法错误" in item for item in result.feedback))

    def test_file_not_exist(self):
        criterion = self._create_criterion("missing.json")
        result = self.checker.validate(criterion, self.workspace_root)
        self.assertFalse(result.passed)

    def test_path_outside_workspace(self):
        criterion = self._create_criterion("../outside.json")
        result = self.checker.validate(criterion, self.workspace_root)
        self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()