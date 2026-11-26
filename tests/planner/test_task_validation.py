
import tempfile
import unittest
from pathlib import Path

from nanobot.planner.models import AcceptanceCriterion, CriterionResult
from nanobot.planner.validation.base import Checker
from nanobot.planner.validation.registry import CheckerRegistry, create_default_registry
from nanobot.planner.validation.validator import TaskResultValidator
from nanobot.planner.validation.checkers.file_system import (
    MAX_FILE_SIZE,
    DirectoryExistsChecker,
    FileContainsChecker,
    FileExistsChecker,
)
from nanobot.planner.validation.checkers.code import (
    JsonValidChecker,
    PythonSyntaxChecker,
)


class TestCheckerRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = CheckerRegistry()

    def test_register_and_get_checker(self):
        checker = FileExistsChecker()
        self.registry.register("file_exists", checker)
        self.assertIs(self.registry.get_checker("file_exists"), checker)

    def test_has_checker(self):
        self.assertFalse(self.registry.has_checker("file_exists"))
        self.registry.register("file_exists", FileExistsChecker())
        self.assertTrue(self.registry.has_checker("file_exists"))

    def test_register_duplicate_checker_raises_error(self):
        self.registry.register("file_exists", FileExistsChecker())
        with self.assertRaises(ValueError):
            self.registry.register("file_exists", FileExistsChecker())

    def test_get_unregistered_checker_raises_error(self):
        with self.assertRaises(ValueError):
            self.registry.get_checker("unknown")

    def test_create_default_registry_registers_all_checkers(self):
        registry = create_default_registry()
        expected_checkers = {
            "file_exists": FileExistsChecker,
            "directory_exists": DirectoryExistsChecker,
            "file_contains": FileContainsChecker,
            "python_syntax": PythonSyntaxChecker,
            "json_valid": JsonValidChecker,
        }
        for criterion_type, checker_class in expected_checkers.items():
            with self.subTest(criterion_type=criterion_type):
                self.assertIsInstance(
                    registry.get_checker(criterion_type),
                    checker_class,
                )


class TestFileExistsChecker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workspace_root = Path(self.temp_dir.name)
        self.checker = FileExistsChecker()

    def _criterion(self, path):
        return AcceptanceCriterion(
            criterion_id="C1",
            type="file_exists",
            description="检查文件是否存在",
            parameters={"path": path},
        )

    def test_existing_file_passes(self):
        target = self.workspace_root / "test.txt"
        target.write_text("hello", encoding="utf-8")
        result = self.checker.validate(
            self._criterion("test.txt"),
            self.workspace_root,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.criterion_id, "C1")
        self.assertTrue(result.feedback)

    def test_missing_file_fails(self):
        result = self.checker.validate(
            self._criterion("missing.txt"),
            self.workspace_root,
        )
        self.assertFalse(result.passed)
        self.assertTrue(result.feedback)

    def test_directory_is_not_a_file(self):
        (self.workspace_root / "folder").mkdir()
        result = self.checker.validate(
            self._criterion("folder"),
            self.workspace_root,
        )
        self.assertFalse(result.passed)

    def test_invalid_path_fails(self):
        invalid_paths = [None, "", "   ", 123]
        for path in invalid_paths:
            with self.subTest(path=path):
                result = self.checker.validate(
                    self._criterion(path),
                    self.workspace_root,
                )
                self.assertFalse(result.passed)

    def test_path_traversal_fails(self):
        result = self.checker.validate(
            self._criterion("../outside.txt"),
            self.workspace_root,
        )
        self.assertFalse(result.passed)
        self.assertIn("超出工作空间范围", result.feedback[0])


class TestDirectoryExistsChecker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workspace_root = Path(self.temp_dir.name)
        self.checker = DirectoryExistsChecker()

    def _criterion(self, path):
        return AcceptanceCriterion(
            criterion_id="C1",
            type="directory_exists",
            description="检查目录是否存在",
            parameters={"path": path},
        )

    def test_existing_directory_passes(self):
        (self.workspace_root / "folder").mkdir()
        result = self.checker.validate(
            self._criterion("folder"),
            self.workspace_root,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.criterion_id, "C1")

    def test_missing_directory_fails(self):
        result = self.checker.validate(
            self._criterion("missing"),
            self.workspace_root,
        )
        self.assertFalse(result.passed)

    def test_file_is_not_a_directory(self):
        (self.workspace_root / "test.txt").write_text(
            "hello",
            encoding="utf-8",
        )
        result = self.checker.validate(
            self._criterion("test.txt"),
            self.workspace_root,
        )
        self.assertFalse(result.passed)

    def test_invalid_path_fails(self):
        invalid_paths = [None, "", "   ", 123]
        for path in invalid_paths:
            with self.subTest(path=path):
                result = self.checker.validate(
                    self._criterion(path),
                    self.workspace_root,
                )
                self.assertFalse(result.passed)

    def test_path_traversal_fails(self):
        result = self.checker.validate(
            self._criterion("../outside"),
            self.workspace_root,
        )
        self.assertFalse(result.passed)
        self.assertIn("超出工作空间范围", result.feedback[0])


class TestFileContainsChecker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workspace_root = Path(self.temp_dir.name)
        self.checker = FileContainsChecker()

    def _criterion(self, path, text):
        return AcceptanceCriterion(
            criterion_id="C1",
            type="file_contains",
            description="检查文件是否包含指定内容",
            parameters={"path": path, "text": text},
        )

    def test_expected_text_exists(self):
        target = self.workspace_root / "test.txt"
        target.write_text("hello nanobot", encoding="utf-8")
        result = self.checker.validate(
            self._criterion("test.txt", "nanobot"),
            self.workspace_root,
        )
        self.assertTrue(result.passed)
        self.assertIn("nanobot", result.feedback[0])

    def test_expected_text_missing(self):
        target = self.workspace_root / "test.txt"
        target.write_text("hello world", encoding="utf-8")
        result = self.checker.validate(
            self._criterion("test.txt", "nanobot"),
            self.workspace_root,
        )
        self.assertFalse(result.passed)
        self.assertIn("不包含期望文本", result.feedback[0])

    def test_missing_file_fails(self):
        result = self.checker.validate(
            self._criterion("missing.txt", "hello"),
            self.workspace_root,
        )
        self.assertFalse(result.passed)

    def test_invalid_expected_text_fails(self):
        target = self.workspace_root / "test.txt"
        target.write_text("hello", encoding="utf-8")
        invalid_texts = [None, 123, "", "   "]
        for text in invalid_texts:
            with self.subTest(text=text):
                result = self.checker.validate(
                    self._criterion("test.txt", text),
                    self.workspace_root,
                )
                self.assertFalse(result.passed)
                self.assertIn("text 参数无效", result.feedback[0])

    def test_file_exceeding_size_limit_fails(self):
        target = self.workspace_root / "large.txt"
        target.write_bytes(b"a" * (MAX_FILE_SIZE + 1))
        result = self.checker.validate(
            self._criterion("large.txt", "a"),
            self.workspace_root,
        )
        self.assertFalse(result.passed)
        self.assertIn("超过 1MB", result.feedback[0])

    def test_invalid_utf8_file_fails(self):
        target = self.workspace_root / "invalid.txt"
        target.write_bytes(b"\xff\xfe\xfa")
        result = self.checker.validate(
            self._criterion("invalid.txt", "hello"),
            self.workspace_root,
        )
        self.assertFalse(result.passed)


class TestPythonSyntaxChecker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workspace_root = Path(self.temp_dir.name)
        self.checker = PythonSyntaxChecker()

    def _criterion(self, path):
        return AcceptanceCriterion(
            criterion_id="C1",
            type="python_syntax",
            description="检查 Python 语法",
            parameters={"path": path},
        )

    def test_valid_python_file_passes(self):
        target = self.workspace_root / "test.py"
        target.write_text(
            "def hello():\n    return 'hello'\n",
            encoding="utf-8",
        )
        result = self.checker.validate(
            self._criterion("test.py"),
            self.workspace_root,
        )
        self.assertTrue(result.passed)
        self.assertIn("语法检查通过", result.feedback[0])

    def test_invalid_python_file_fails(self):
        target = self.workspace_root / "test.py"
        target.write_text(
            "def hello(\n    return 'hello'\n",
            encoding="utf-8",
        )
        result = self.checker.validate(
            self._criterion("test.py"),
            self.workspace_root,
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("语法错误" in item for item in result.feedback))
        self.assertTrue(any("错误位置" in item for item in result.feedback))

    def test_missing_python_file_fails(self):
        result = self.checker.validate(
            self._criterion("missing.py"),
            self.workspace_root,
        )
        self.assertFalse(result.passed)

    def test_python_file_path_traversal_fails(self):
        result = self.checker.validate(
            self._criterion("../outside.py"),
            self.workspace_root,
        )
        self.assertFalse(result.passed)


class TestJsonValidChecker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workspace_root = Path(self.temp_dir.name)
        self.checker = JsonValidChecker()

    def _criterion(self, path):
        return AcceptanceCriterion(
            criterion_id="C1",
            type="json_valid",
            description="检查 JSON 格式",
            parameters={"path": path},
        )

    def test_valid_json_object_passes(self):
        target = self.workspace_root / "test.json"
        target.write_text(
            '{"name": "nanobot", "enabled": true}',
            encoding="utf-8",
        )
        result = self.checker.validate(
            self._criterion("test.json"),
            self.workspace_root,
        )
        self.assertTrue(result.passed)
        self.assertIn("语法检查通过", result.feedback[0])

    def test_valid_json_array_passes(self):
        target = self.workspace_root / "test.json"
        target.write_text('[1, 2, 3]', encoding="utf-8")
        result = self.checker.validate(
            self._criterion("test.json"),
            self.workspace_root,
        )
        self.assertTrue(result.passed)

    def test_invalid_json_fails(self):
        target = self.workspace_root / "test.json"
        target.write_text(
            '{"name": "nanobot",}',
            encoding="utf-8",
        )
        result = self.checker.validate(
            self._criterion("test.json"),
            self.workspace_root,
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("json 语法错误" in item for item in result.feedback))
        self.assertTrue(any("错误位置" in item for item in result.feedback))

    def test_missing_json_file_fails(self):
        result = self.checker.validate(
            self._criterion("missing.json"),
            self.workspace_root,
        )
        self.assertFalse(result.passed)

    def test_json_path_traversal_fails(self):
        result = self.checker.validate(
            self._criterion("../outside.json"),
            self.workspace_root,
        )
        self.assertFalse(result.passed)


class _FakeChecker(Checker):
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def validate(self, criterion, workspace_root):
        if self.error is not None:
            raise self.error
        return self.result


class TestTaskResultValidator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workspace_root = Path(self.temp_dir.name)
        self.registry = create_default_registry()
        self.validator = TaskResultValidator(self.registry)

    def _criterion(self, criterion_id, criterion_type, parameters):
        return AcceptanceCriterion(
            criterion_id=criterion_id,
            type=criterion_type,
            description="测试验收标准",
            parameters=parameters,
        )

    def test_all_criteria_pass(self):
        target = self.workspace_root / "test.txt"
        target.write_text("hello nanobot", encoding="utf-8")
        criteria = [
            self._criterion("C1", "file_exists", {"path": "test.txt"}),
            self._criterion(
                "C2",
                "file_contains",
                {"path": "test.txt", "text": "nanobot"},
            ),
        ]
        result = self.validator.validate(criteria, self.workspace_root)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.criterion_results), 2)
        self.assertEqual(len(result.feedback), 2)
        self.assertTrue(all(item.passed for item in result.criterion_results))

    def test_one_failed_criterion_fails_validation(self):
        target = self.workspace_root / "test.txt"
        target.write_text("hello", encoding="utf-8")
        criteria = [
            self._criterion("C1", "file_exists", {"path": "test.txt"}),
            self._criterion(
                "C2",
                "file_contains",
                {"path": "test.txt", "text": "nanobot"},
            ),
        ]
        result = self.validator.validate(criteria, self.workspace_root)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.criterion_results), 2)
        self.assertTrue(result.criterion_results[0].passed)
        self.assertFalse(result.criterion_results[1].passed)
        self.assertTrue(any("不包含期望文本" in item for item in result.feedback))

    def test_all_failed_criteria_are_collected(self):
        criteria = [
            self._criterion("C1", "file_exists", {"path": "missing1.txt"}),
            self._criterion("C2", "file_exists", {"path": "missing2.txt"}),
        ]
        result = self.validator.validate(criteria, self.workspace_root)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.criterion_results), 2)
        self.assertEqual(len(result.feedback), 2)
        self.assertTrue(all(not item.passed for item in result.criterion_results))

    def test_empty_acceptance_criteria_fails(self):
        result = self.validator.validate([], self.workspace_root)
        self.assertFalse(result.passed)
        self.assertEqual(result.criterion_results, [])
        self.assertTrue(result.feedback)
        self.assertIn("没有配置验收标准", result.feedback[0])

    def test_unknown_checker_type_becomes_failed_result(self):
        criterion = self._criterion("C1", "unknown", {})
        result = self.validator.validate([criterion], self.workspace_root)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.criterion_results), 1)
        self.assertFalse(result.criterion_results[0].passed)
        self.assertIn("验收检查发生异常", result.feedback[0])

    def test_checker_exception_becomes_failed_result(self):
        self.registry.register(
            "custom",
            _FakeChecker(error=RuntimeError("checker crashed")),
        )
        validator = TaskResultValidator(self.registry)
        criterion = self._criterion("C1", "custom", {})
        result = validator.validate([criterion], self.workspace_root)
        self.assertFalse(result.passed)
        self.assertIn("checker crashed", result.feedback[0])

    def test_checker_returning_wrong_object_becomes_failed_result(self):
        self.registry.register("custom", _FakeChecker(result="invalid"))
        validator = TaskResultValidator(self.registry)
        criterion = self._criterion("C1", "custom", {})
        result = validator.validate([criterion], self.workspace_root)
        self.assertFalse(result.passed)
        self.assertIn("返回结果格式不正确", result.feedback[0])

    def test_checker_returning_wrong_criterion_id_becomes_failed_result(self):
        self.registry.register(
            "custom",
            _FakeChecker(
                result=CriterionResult(
                    criterion_id="WRONG",
                    passed=True,
                    feedback=["ok"],
                )
            ),
        )
        validator = TaskResultValidator(self.registry)
        criterion = self._criterion("C1", "custom", {})
        result = validator.validate([criterion], self.workspace_root)
        self.assertFalse(result.passed)
        self.assertIn("criterion_id", result.feedback[0])

    def test_checker_returning_non_boolean_passed_becomes_failed_result(self):
        self.registry.register(
            "custom",
            _FakeChecker(
                result=CriterionResult(
                    criterion_id="C1",
                    passed=1,
                    feedback=[],
                )
            ),
        )
        validator = TaskResultValidator(self.registry)
        criterion = self._criterion("C1", "custom", {})
        result = validator.validate([criterion], self.workspace_root)
        self.assertFalse(result.passed)
        self.assertIn("passed必须是布尔值", result.feedback[0])

    def test_checker_returning_non_list_feedback_becomes_failed_result(self):
        self.registry.register(
            "custom",
            _FakeChecker(
                result=CriterionResult(
                    criterion_id="C1",
                    passed=True,
                    feedback="invalid",
                )
            ),
        )
        validator = TaskResultValidator(self.registry)
        criterion = self._criterion("C1", "custom", {})
        result = validator.validate([criterion], self.workspace_root)
        self.assertFalse(result.passed)
        self.assertIn("feedback必须是字符串列表", result.feedback[0])

    def test_checker_returning_non_string_feedback_item_becomes_failed_result(self):
        self.registry.register(
            "custom",
            _FakeChecker(
                result=CriterionResult(
                    criterion_id="C1",
                    passed=True,
                    feedback=["ok", 123],
                )
            ),
        )
        validator = TaskResultValidator(self.registry)
        criterion = self._criterion("C1", "custom", {})
        result = validator.validate([criterion], self.workspace_root)
        self.assertFalse(result.passed)
        self.assertIn("feedback必须是字符串列表", result.feedback[0])

    def test_validation_preserves_criterion_order(self):
        target = self.workspace_root / "test.txt"
        target.write_text("hello", encoding="utf-8")
        criteria = [
            self._criterion("C2", "file_exists", {"path": "test.txt"}),
            self._criterion("C1", "file_exists", {"path": "missing.txt"}),
        ]
        result = self.validator.validate(criteria, self.workspace_root)
        self.assertEqual(
            [item.criterion_id for item in result.criterion_results],
            ["C2", "C1"],
        )


if __name__ == "__main__":
    unittest.main()
