
import json
import unittest
from unittest.mock import MagicMock
from nanobot.planner.planner import Planner


class TestPlannerDiagnostics(unittest.TestCase):
    def setUp(self):
        self.planner = Planner(MagicMock())

    def _build_response(self, criterion=None):
        if criterion is None:
            criterion = {
                "criterion_id": "C1",
                "type": "file_exists",
                "description": "检查文件是否存在",
                "parameters": {"path": "app.py"}
            }
        return json.dumps({
            "tasks": [
                {
                    "task_id": "T1",
                    "description": "创建应用文件",
                    "dependencies": [],
                    "acceptance_criteria": [criterion]
                }
            ]
        })

    def test_invalid_json_reports_json_error(self):
        with self.assertRaisesRegex(ValueError, "JSON解析失败"):
            self.planner._parse_response('{"tasks": [')

    def test_tasks_type_error_is_specific(self):
        with self.assertRaisesRegex(ValueError, "tasks必须是数组"):
            self.planner._parse_response('{"tasks": {}}')

    def test_empty_task_id_reports_task_location(self):
        data = json.loads(self._build_response())
        data["tasks"][0]["task_id"] = ""
        with self.assertRaisesRegex(ValueError, "第1个任务.*task_id.*不能为空"):
            self.planner._parse_response(json.dumps(data))

    def test_empty_task_description_reports_task_location(self):
        data = json.loads(self._build_response())
        data["tasks"][0]["description"] = ""
        with self.assertRaisesRegex(ValueError, "T1.*description.*不能为空"):
            self.planner._parse_response(json.dumps(data))

    def test_empty_criterion_id_reports_criterion_location(self):
        criterion = {
            "criterion_id": "",
            "type": "file_exists",
            "description": "检查文件是否存在",
            "parameters": {"path": "app.py"}
        }
        with self.assertRaisesRegex(ValueError, "T1.*第1条.*criterion_id.*不能为空"):
            self.planner._parse_response(self._build_response(criterion))

    def test_unsupported_criterion_type_reports_type(self):
        criterion = {
            "criterion_id": "C1",
            "type": "shell_command",
            "description": "执行命令",
            "parameters": {"path": "app.py"}
        }
        with self.assertRaisesRegex(ValueError, "T1/C1.*不支持.*shell_command"):
            self.planner._parse_response(self._build_response(criterion))

    def test_missing_parameter_reports_parameter_name(self):
        criterion = {
            "criterion_id": "C1",
            "type": "file_contains",
            "description": "检查文件内容",
            "parameters": {"path": "app.py"}
        }
        with self.assertRaisesRegex(ValueError, "缺少参数：text"):
            self.planner._parse_response(self._build_response(criterion))

    def test_extra_parameter_reports_parameter_name(self):
        criterion = {
            "criterion_id": "C1",
            "type": "file_exists",
            "description": "检查文件是否存在",
            "parameters": {
                "path": "app.py",
                "extra": "value"
            }
        }
        with self.assertRaisesRegex(ValueError, "不允许的参数：extra"):
            self.planner._parse_response(self._build_response(criterion))

    def test_empty_criterion_description_reports_location(self):
        criterion = {
            "criterion_id": "C1",
            "type": "file_exists",
            "description": "",
            "parameters": {"path": "app.py"}
        }
        with self.assertRaisesRegex(ValueError, "T1/C1.*description.*不能为空"):
            self.planner._parse_response(self._build_response(criterion))

    def test_valid_response_still_passes(self):
        tasks = self.planner._parse_response(self._build_response())
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_id"], "T1")
        self.assertEqual(tasks[0]["acceptance_criteria"][0].criterion_id, "C1")


if __name__ == "__main__":
    unittest.main()
