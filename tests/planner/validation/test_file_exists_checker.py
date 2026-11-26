
import tempfile
import unittest
from pathlib import Path

from nanobot.planner.models import AcceptanceCriterion, CriterionResult
from nanobot.planner.validation.checkers.file_system import FileExistsChecker


class TestFileExistsChecker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp_dir.name)
        self.checker = FileExistsChecker()

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_criterion(self, path):
        return AcceptanceCriterion(
            criterion_id="C1",
            type="file_exists",
            description="检查文件是否存在",
            parameters={"path": path},
        )

    def test_existing_file_should_pass(self):
        target = self.workspace_root / "test.txt"
        target.write_text("hello", encoding="utf-8")

        result = self.checker.validate(
            self.create_criterion("test.txt"),
            self.workspace_root,
        )

        self.assertIsInstance(result, CriterionResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.criterion_id, "C1")
        self.assertTrue(result.feedback)
        self.assertIn(str(target.resolve()), result.evidence)

    def test_nonexistent_file_should_fail(self):
        result = self.checker.validate(
            self.create_criterion("missing.txt"),
            self.workspace_root,
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.criterion_id, "C1")

    def test_directory_should_fail(self):
        (self.workspace_root / "mydir").mkdir()

        result = self.checker.validate(
            self.create_criterion("mydir"),
            self.workspace_root,
        )

        self.assertFalse(result.passed)

    def test_empty_path_should_fail(self):
        result = self.checker.validate(
            self.create_criterion(""),
            self.workspace_root,
        )

        self.assertFalse(result.passed)

    def test_whitespace_path_should_fail(self):
        result = self.checker.validate(
            self.create_criterion("   "),
            self.workspace_root,
        )

        self.assertFalse(result.passed)

    def test_non_string_path_should_fail(self):
        result = self.checker.validate(
            self.create_criterion(123),
            self.workspace_root,
        )

        self.assertFalse(result.passed)

    def test_parent_directory_traversal_should_fail(self):
        outside_file = Path(self.temp_dir.name).parent / "outside_test_file.txt"
        outside_file.write_text("outside", encoding="utf-8")

        try:
            result = self.checker.validate(
                self.create_criterion("../outside_test_file.txt"),
                self.workspace_root,
            )

            self.assertFalse(result.passed)
        finally:
            outside_file.unlink(missing_ok=True)

    def test_absolute_path_outside_workspace_should_fail(self):
        outside_file = Path(self.temp_dir.name).parent / "outside_absolute_test.txt"
        outside_file.write_text("outside", encoding="utf-8")

        try:
            result = self.checker.validate(
                self.create_criterion(str(outside_file)),
                self.workspace_root,
            )

            self.assertFalse(result.passed)
        finally:
            outside_file.unlink(missing_ok=True)

    def test_symlink_outside_workspace_should_fail(self):
        outside_dir = Path(tempfile.mkdtemp())
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("secret", encoding="utf-8")

        link_path = self.workspace_root / "external_link"

        try:
            try:
                link_path.symlink_to(outside_file)
            except (OSError, NotImplementedError):
                self.skipTest("当前系统不支持创建符号链接")

            result = self.checker.validate(
                self.create_criterion("external_link"),
                self.workspace_root,
            )

            self.assertFalse(result.passed)
        finally:
            import shutil
            shutil.rmtree(outside_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
