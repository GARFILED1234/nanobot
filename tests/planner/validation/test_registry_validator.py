import tempfile
import unittest
from pathlib import Path

from nanobot.planner.models import (
    AcceptanceCriterion,
    CriterionResult,
    ValidationResult,
)
from nanobot.planner.validation.base import Checker
from nanobot.planner.validation.registry import (
    CheckerRegistry,
    create_default_registry,
)
from nanobot.planner.validation.validator import TaskResultValidator


class MockChecker(Checker):
    def __init__(self, passed=True):
        self.passed = passed

    def validate(self, criterion, workspace_root):
        return CriterionResult(
            criterion_id=criterion.criterion_id,
            passed=self.passed,
            feedback=["模拟检查结果"],
        )


class ExceptionChecker(Checker):
    def validate(self, criterion, workspace_root):
        raise RuntimeError("模拟 Checker 异常")


class TestCheckerRegistry(unittest.TestCase):
    def test_register_checker(self):
        registry = CheckerRegistry()
        checker = MockChecker()
        registry.register("test", checker)
        self.assertTrue(registry.has_checker("test"))
        self.assertIs(registry.get_checker("test"), checker)

    def test_register_duplicate_checker(self):
        registry = CheckerRegistry()
        registry.register("test", MockChecker())
        with self.assertRaises(ValueError):
            registry.register("test", MockChecker())

    def test_get_missing_checker(self):
        registry = CheckerRegistry()
        with self.assertRaises(ValueError):
            registry.get_checker("missing")

    def test_default_registry(self):
        registry = create_default_registry()
        expected_types = [
            "file_exists",
            "directory_exists",
            "file_contains",
            "python_syntax",
            "json_valid",
        ]
        for criterion_type in expected_types:
            with self.subTest(criterion_type=criterion_type):
                self.assertTrue(registry.has_checker(criterion_type))


class TestTaskResultValidator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace_root = Path(self.temp_dir.name)
        self.registry = CheckerRegistry()
        self.validator = TaskResultValidator(self.registry)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_criterion(self, criterion_id="C1", criterion_type="test"):
        return AcceptanceCriterion(
            criterion_id=criterion_id,
            type=criterion_type,
            description="测试验收标准",
            parameters={},
        )

    def test_all_criteria_passed(self):
        self.registry.register("test", MockChecker(True))
        criteria = [
            self._create_criterion("C1"),
            self._create_criterion("C2"),
        ]
        result = self.validator.validate(criteria, self.workspace_root)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.criterion_results), 2)

    def test_one_criterion_failed(self):
        self.registry.register("pass", MockChecker(True))
        self.registry.register("fail", MockChecker(False))
        criteria = [
            self._create_criterion("C1", "pass"),
            self._create_criterion("C2", "fail"),
        ]
        result = self.validator.validate(criteria, self.workspace_root)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.criterion_results), 2)

    def test_empty_acceptance_criteria(self):
        result = self.validator.validate([], self.workspace_root)
        self.assertFalse(result.passed)
        self.assertEqual(result.criterion_results, [])
        self.assertTrue(result.feedback)

    def test_checker_exception(self):
        self.registry.register("exception", ExceptionChecker())
        criteria = [
            self._create_criterion("C1", "exception"),
        ]
        result = self.validator.validate(criteria, self.workspace_root)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.criterion_results), 1)
        self.assertTrue(any("模拟 Checker 异常" in item for item in result.feedback))

    def test_unknown_checker_type(self):
        criteria = [
            self._create_criterion("C1", "unknown"),
        ]
        result = self.validator.validate(criteria, self.workspace_root)
        self.assertFalse(result.passed)
        self.assertEqual(len(result.criterion_results), 1)


if __name__ == "__main__":
    unittest.main()