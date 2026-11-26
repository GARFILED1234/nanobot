
import tempfile
import unittest
import shutil
from pathlib import Path

from nanobot.planner.models import AcceptanceCriterion, CriterionResult
from nanobot.planner.validation.checkers.file_system import DirectoryExistsChecker


class TestDirectoryExistsChecker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.workspace_root = self.base_dir / "workspace"
        self.workspace_root.mkdir()
        self.checker = DirectoryExistsChecker()

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_criterion(self, path):
        return AcceptanceCriterion(
            criterion_id="C2",
            type="directory_exists",
            description="检查目录是否存在",
            parameters={"path": path},
        )

    def test_existing_directory_should_pass(self):
        target = self.workspace_root / "frontend"
        target.mkdir()

        result = self.checker.validate(
            self.create_criterion("frontend"),
            self.workspace_root,
        )

        self.assertIsInstance(result, CriterionResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.criterion_id, "C2")
        self.assertTrue(result.feedback)
        self.assertIn(str(target.resolve()), result.evidence)

    def test_nonexistent_directory_should_fail(self):
        result = self.checker.validate(
            self.create_criterion("missing"),
            self.workspace_root,
        )

        self.assertFalse(result.passed)

    def test_file_should_fail(self):
        target = self.workspace_root / "test.txt"
        target.write_text("hello", encoding="utf-8")

        result = self.checker.validate(
            self.create_criterion("test.txt"),
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
        outside_dir = self.base_dir / "outside"
        outside_dir.mkdir()

        result = self.checker.validate(
            self.create_criterion("../outside"),
            self.workspace_root,
        )

        self.assertFalse(result.passed)

    def test_absolute_path_outside_workspace_should_fail(self):
        outside_dir = self.base_dir / "outside"
        outside_dir.mkdir()

        result = self.checker.validate(
            self.create_criterion(str(outside_dir)),
            self.workspace_root,
        )

        self.assertFalse(result.passed)

    def test_symlink_outside_workspace_should_fail(self):
        outside_dir = self.base_dir / "outside"
        outside_dir.mkdir()

        link_path = self.workspace_root / "external_link"

        try:
            link_path.symlink_to(outside_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("当前系统不支持创建符号链接")

        result = self.checker.validate(
            self.create_criterion("external_link"),
            self.workspace_root,
        )

        self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()
