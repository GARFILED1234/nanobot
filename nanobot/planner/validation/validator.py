
from pathlib import Path
from ..models import AcceptanceCriterion, CriterionResult, ValidationResult
from .registry import CheckerRegistry


class TaskResultValidator:
    def __init__(self, checker_registry: CheckerRegistry):
        self.checker_registry = checker_registry

    def validate(self, acceptance_criteria: list[AcceptanceCriterion], workspace_root: Path) -> ValidationResult:
        passed = True
        criterion_results = []
        feedback = []
        if not acceptance_criteria:
            return ValidationResult(
                passed=False,
                feedback=["当前任务没有配置验收标准，无法确认任务是否通过"],
                criterion_results=[]
            )
        for criterion in acceptance_criteria:
            try:
                checker = self.checker_registry.get_checker(criterion.type)
                criterion_result = checker.validate(criterion, workspace_root)
                if not isinstance(criterion_result, CriterionResult):
                    raise ValueError("Checker返回结果格式不正确")
                if criterion_result.criterion_id != criterion.criterion_id:
                    raise ValueError("Checker返回的criterion_id与验收标准不一致")
                if not isinstance(criterion_result.passed, bool):
                    raise ValueError("Checker返回的passed必须是布尔值")
                if not isinstance(criterion_result.feedback, list) or not all(isinstance(item, str) for item in criterion_result.feedback):
                    raise ValueError("Checker返回的feedback必须是字符串列表")
            except Exception as err:
                criterion_result = CriterionResult(
                    criterion_id=criterion.criterion_id,
                    passed=False,
                    feedback=[f"验收检查发生异常：{err}"]
                )
            criterion_results.append(criterion_result)
            if not criterion_result.passed:
                passed = False
            feedback.extend(criterion_result.feedback)
        return ValidationResult(
            passed=passed,
            feedback=feedback,
            criterion_results=criterion_results
        )
