
import tempfile
import unittest
from pathlib import Path

from nanobot.planner.models import AcceptanceCriterion
from nanobot.planner.validation.checkers.file_system import FileContainsChecker


class TestFileContainsChecker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp_dir.name)
        self.checker = FileContainsChecker()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _criterion(self, path="test.txt", text="hello"):
        return AcceptanceCriterion(
            criterion_id="C1",
            type="file_contains",
            description="检查文件是否包含指定文本",
            parameters={
                "path": path,
                "text": text,
            },
        )

    def _validate(self, path="test.txt", text="hello"):
        criterion = self._criterion(path, text)
        return self.checker.validate(criterion, self.workspace_root)

    def test_file_contains_text(self):
        target = self.workspace_root / "test.txt"
        target.write_text("hello world", encoding="utf-8")

        result = self._validate()

        self.assertTrue(result.passed)
        self.assertEqual(result.criterion_id, "C1")
        self.assertIn("hello", result.evidence[0])
        self.assertEqual(result.feedback, "文件包含期望文本")

    def test_file_does_not_contain_text(self):
        target = self.workspace_root / "test.txt"
        target.write_text("goodbye world", encoding="utf-8")

        result = self._validate()

        self.assertFalse(result.passed)
        self.assertEqual(result.feedback, "文件内容不符合预期要求")

    def test_file_not_exists(self):
        result = self._validate("missing.txt", "hello")

        self.assertFalse(result.passed)

    def test_target_is_directory(self):
        (self.workspace_root / "test.txt").mkdir()

        result = self._validate()

        self.assertFalse(result.passed)
        self.assertEqual(result.feedback, "文件检查失败")

    def test_empty_path(self):
        result = self._validate("", "hello")

        self.assertFalse(result.passed)

    def test_whitespace_path(self):
        result = self._validate("   ", "hello")

        self.assertFalse(result.passed)

    def test_non_string_path(self):
        result = self._validate(123, "hello")

        self.assertFalse(result.passed)

    def test_empty_text(self):
        target = self.workspace_root / "test.txt"
        target.write_text("hello world", encoding="utf-8")

        result = self._validate("test.txt", "")

        self.assertFalse(result.passed)

    def test_whitespace_text(self):
        target = self.workspace_root / "test.txt"
        target.write_text("hello world", encoding="utf-8")

        result = self._validate("test.txt", "   ")

        self.assertFalse(result.passed)

    def test_non_string_text(self):
        target = self.workspace_root / "test.txt"
        target.write_text("hello world", encoding="utf-8")

        result = self._validate("test.txt", 123)

        self.assertFalse(result.passed)

    def test_path_traversal(self):
        outside_file = self.workspace_root.parent / "outside.txt"
        outside_file.write_text("hello", encoding="utf-8")

        try:
            result = self._validate("../outside.txt", "hello")
            self.assertFalse(result.passed)
        finally:
            outside_file.unlink(missing_ok=True)

    def test_absolute_path_outside_workspace(self):
        outside_file = self.workspace_root.parent / "outside.txt"
        outside_file.write_text("hello", encoding="utf-8")

        try:
            result = self._validate(str(outside_file), "hello")
            self.assertFalse(result.passed)
        finally:
            outside_file.unlink(missing_ok=True)

    def test_file_exceeds_one_mb(self):
        target = self.workspace_root / "large.txt"
        target.write_bytes(b"a" * (1024 * 1024 + 1))

        result = self._validate("large.txt", "hello")

        self.assertFalse(result.passed)
        self.assertEqual(result.feedback, "文件大小超过1MB，跳过检查")

    def test_invalid_utf8_file(self):
        target = self.workspace_root / "invalid.txt"
        target.write_bytes(b"\xff\xfe\xfa")

        result = self._validate("invalid.txt", "hello")

        self.assertFalse(result.passed)

    def test_leading_and_trailing_whitespace_in_text(self):
        target = self.workspace_root / "test.txt"
        target.write_text("hello world", encoding="utf-8")

        result = self._validate("test.txt", "  hello world  ")

        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
