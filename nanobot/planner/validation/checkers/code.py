
from pathlib import Path
from ..base import Checker
from ...models import AcceptanceCriterion, CriterionResult
from .file_system import read_text_file
import ast
import json

class PythonSyntaxChecker(Checker):
    def validate(self, criterion, workspace_root):
        criterion_id = criterion.criterion_id
        passed = True
        feedback = []
        relative_path = criterion.parameters.get("path")
        r_dict = read_text_file(relative_path, workspace_root)
        passed = r_dict["passed"]
        if not passed:
            feedback.append(r_dict["feedback"])
            return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)
        content = r_dict["content"]
        try:
            ast.parse(content, filename=relative_path)
        except SyntaxError as err:
            passed = False
            feedback.append(f"Python 语法错误：{err.msg}")
            feedback.append(f"错误位置：第 {err.lineno} 行，第 {err.offset} 列")
            return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)
        feedback.append(f"Python 文件语法检查通过：{relative_path}")
        return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)

class JsonValidChecker(Checker):
    def validate(self, criterion, workspace_root):
        criterion_id = criterion.criterion_id
        passed = True
        feedback = []
        relative_path = criterion.parameters.get("path")
        r_dict = read_text_file(relative_path, workspace_root)
        passed = r_dict["passed"]
        if not passed:
            feedback.append(r_dict["feedback"])
            return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)
        content = r_dict["content"]
        try:
            json.loads(content)
        except json.JSONDecodeError as err:
            passed = False
            feedback.append(f"json 语法错误：{err.msg}")
            feedback.append(f"错误位置：第 {err.lineno} 行，第 {err.colno} 列")
            return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)
        feedback.append(f"json 文件语法检查通过：{relative_path}")
        return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)