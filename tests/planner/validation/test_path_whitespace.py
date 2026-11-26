
import tempfile
import unittest
from pathlib import Path

from nanobot.planner.validation.checkers.file_system import _is_path_exist


class TestPathWhitespace(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normal_path_is_accepted(self):
        result = _is_path_exist("app.py", self.workspace_root)
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["target_path"],
            (self.workspace_root / "app.py").resolve(),
        )

    def test_internal_spaces_are_preserved(self):
        project_dir = self.workspace_root / "my project"
        project_dir.mkdir()
        result = _is_path_exist("my project/app.py", self.workspace_root)
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["target_path"],
            (project_dir / "app.py").resolve(),
        )

    def test_leading_or_trailing_whitespace_raises_value_error(self):
        paths = [
            " app.py",
            "app.py ",
            " app.py ",
            "\tapp.py",
            "app.py\n",
        ]
        for path in paths:
            with self.subTest(path=repr(path)):
                with self.assertRaisesRegex(
                    ValueError,
                    "文件路径不能包含首尾空格",
                ):
                    _is_path_exist(path, self.workspace_root)

    def test_whitespace_only_path_returns_failure(self):
        result = _is_path_exist("   ", self.workspace_root)
        self.assertFalse(result["passed"])
        self.assertEqual(result["feedback"], "文件路径不存在")

    def test_path_traversal_is_rejected(self):
        result = _is_path_exist("../outside.py", self.workspace_root)
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["feedback"],
            "文件路径超出工作空间范围",
        )


if __name__ == "__main__":
    unittest.main()
