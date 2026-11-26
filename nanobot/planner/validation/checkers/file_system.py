
from pathlib import Path
from ..base import Checker
from ...models import AcceptanceCriterion, CriterionResult

MAX_FILE_SIZE = 1024 * 1024


class FileExistsChecker(Checker):
    def validate(self, criterion, workspace_root):
        criterion_id = criterion.criterion_id
        feedback = []
        relative_path = criterion.parameters.get("path")
        r_dict = _is_path_exist(relative_path, workspace_root)
        passed = r_dict["passed"]
        if not passed:
            feedback.append(r_dict["feedback"])
            return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)
        target_path = r_dict.get("target_path")
        try:
            if not target_path.exists() or not target_path.is_file():
                passed = False
                feedback.append("文件检查失败")
                return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)
        except (OSError, ValueError, RuntimeError) as err:
            passed = False
            feedback.append(str(err))
            return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)
        feedback.append(f"文件路径存在且对应位置是一个普通文件：{target_path}")
        return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)


class DirectoryExistsChecker(Checker):
    def validate(self, criterion, workspace_root):
        criterion_id = criterion.criterion_id
        feedback = []
        relative_path = criterion.parameters.get("path")
        r_dict = _is_path_exist(relative_path, workspace_root)
        passed = r_dict["passed"]
        if not passed:
            feedback.append(r_dict["feedback"])
            return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)
        target_path = r_dict.get("target_path")
        try:
            if not target_path.exists() or not target_path.is_dir():
                passed = False
                feedback.append("目录检查失败")
                return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)
        except (OSError, ValueError, RuntimeError) as err:
            passed = False
            feedback.append(str(err))
            return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)
        feedback.append(f"目录路径存在且对应位置是一个普通目录：{target_path}")
        return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)


class FileContainsChecker(Checker):
    def validate(self, criterion, workspace_root):
        criterion_id = criterion.criterion_id
        feedback = []
        relative_path = criterion.parameters.get("path")
        r_dict = read_text_file(relative_path, workspace_root)
        passed = r_dict["passed"]
        if not passed:
            feedback.append(r_dict["feedback"])
            return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)
        pas = criterion.parameters
        etext = pas.get("text")
        if not isinstance(etext, str) or not (etext := etext.strip()):
            passed = False
            feedback.append("验收标准中的 text 参数无效")
            return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)
        content = r_dict["content"]
        if etext not in content:
            passed = False
            feedback.append(f"文件内容不包含期望文本：{etext}")
            return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)
        feedback.append(f"文件包含期望文本：{etext}")
        return CriterionResult(criterion_id=criterion_id, passed=passed, feedback=feedback)


def _is_path_exist(relative_path: str, workspace_root: Path):
    r_dict = {}
    if not isinstance(relative_path, str) or not relative_path.strip():
        r_dict["passed"] = False
        r_dict["feedback"] = "文件路径不存在"
        return r_dict
    if relative_path != relative_path.strip():
        raise ValueError("文件路径不能包含首尾空格")
    try:
        root = workspace_root.resolve()
    except (OSError, ValueError, RuntimeError) as err:
        r_dict["passed"] = False
        r_dict["feedback"] = str(err)
        return r_dict
    try:
        target_path = (root / relative_path).resolve()
    except (OSError, ValueError, RuntimeError) as err:
        r_dict["passed"] = False
        r_dict["feedback"] = str(err)
        return r_dict
    r_dict["target_path"] = target_path
    if not target_path.is_relative_to(root):
        r_dict["passed"] = False
        r_dict["feedback"] = "文件路径超出工作空间范围"
        return r_dict
    r_dict["passed"] = True
    return r_dict


def read_text_file(relative_path: str, workspace_root: Path):
    r_dict = {}
    r1_dict = _is_path_exist(relative_path, workspace_root)
    passed = r1_dict["passed"]
    if not passed:
        r_dict["feedback"] = r1_dict["feedback"]
        r_dict["passed"] = passed
        return r_dict
    target_path = r1_dict.get("target_path")
    try:
        if not target_path.exists() or not target_path.is_file():
            r_dict["feedback"] = "文件检查失败"
            r_dict["passed"] = False
            return r_dict
    except (OSError, ValueError, RuntimeError) as err:
        r_dict["passed"] = False
        r_dict["feedback"] = str(err)
        return r_dict
    try:
        f_size = target_path.stat().st_size
    except (OSError, ValueError, RuntimeError) as err:
        r_dict["passed"] = False
        r_dict["feedback"] = str(err)
        return r_dict
    if f_size > MAX_FILE_SIZE:
        r_dict["feedback"] = f"文件大小为 {f_size} 字节，超过 1MB，跳过检查"
        r_dict["passed"] = False
        return r_dict
    try:
        content = target_path.read_text(encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as err:
        r_dict["passed"] = False
        r_dict["feedback"] = str(err)
        return r_dict
    r_dict["content"] = content
    r_dict["passed"] = True
    return r_dict
