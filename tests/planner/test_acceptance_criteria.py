
import json
import unittest
from unittest.mock import MagicMock
from nanobot.planner.planner import Planner


class TestAcceptanceCriteria(unittest.TestCase):
    def setUp(self):
        self.planner = Planner(MagicMock())
    def _build_response(self, criteria=None, parameters=None, criterion_type="file_exists"):
        if criteria is None:
            criteria = [
                {
                    "criterion_id": "C1",
                    "type": criterion_type,
                    "description": "检查文件是否存在",
                    "parameters": parameters if parameters is not None else {"path": "app.py"}
                }
            ]
        return json.dumps({
            "tasks": [
                {
                    "task_id": "T1",
                    "description": "创建应用文件",
                    "dependencies": [],
                    "acceptance_criteria": criteria
                }
            ]
        })
    def test_empty_acceptance_criteria_rejected(self):
        response = self._build_response(criteria=[])
        with self.assertRaisesRegex(ValueError, "任务验收标准不能为空"):
            self.planner._parse_response(response)
    def test_parameter_name_with_surrounding_spaces_normalized(self):
        response = self._build_response(parameters={" path ": "my project/app.py"})
        tasks = self.planner._parse_response(response)
        parameters = tasks[0]["acceptance_criteria"][0].parameters
        self.assertEqual(parameters, {"path": "my project/app.py"})
    def test_normalized_duplicate_parameter_names_rejected(self):
        response = self._build_response(
            parameters={
                "path": "app.py",
                " path ": "other.py"
            }
        )
        with self.assertRaisesRegex(ValueError, "验收标准参数名称重复"):
            self.planner._parse_response(response)
    def test_path_value_internal_spaces_preserved(self):
        response = self._build_response(parameters={"path": "my project/app.py"})
        tasks = self.planner._parse_response(response)
        parameters = tasks[0]["acceptance_criteria"][0].parameters
        self.assertEqual(parameters["path"], "my project/app.py")
    def test_unsupported_criterion_type_rejected(self):
        response = self._build_response(criterion_type="shell_command")
        with self.assertRaises(ValueError):
            self.planner._parse_response(response)
    def test_missing_required_parameter_rejected(self):
        response = self._build_response(parameters={})
        with self.assertRaisesRegex(ValueError, "验收标准参数不符合要求"):
            self.planner._parse_response(response)
    def test_extra_parameter_rejected(self):
        response = self._build_response(
            parameters={"path": "app.py", "extra": "value"}
        )
        with self.assertRaisesRegex(ValueError, "验收标准参数不符合要求"):
            self.planner._parse_response(response)
    def test_duplicate_criterion_id_after_strip_rejected(self):
        criteria = [
            {
                "criterion_id": "C1",
                "type": "file_exists",
                "description": "检查文件一",
                "parameters": {"path": "app.py"}
            },
            {
                "criterion_id": " C1 ",
                "type": "file_exists",
                "description": "检查文件二",
                "parameters": {"path": "other.py"}
            }
        ]
        response = self._build_response(criteria=criteria)
        with self.assertRaises(ValueError):
            self.planner._parse_response(response)
    def test_duplicate_task_id_rejected(self):
        data = {
            "tasks": [
                {
                    "task_id": "T1",
                    "description": "任务一",
                    "dependencies": [],
                    "acceptance_criteria": [
                        {
                            "criterion_id": "C1",
                            "type": "file_exists",
                            "description": "检查文件",
                            "parameters": {"path": "a.py"}
                        }
                    ]
                },
                {
                    "task_id": " T1 ",
                    "description": "任务二",
                    "dependencies": [],
                    "acceptance_criteria": [
                        {
                            "criterion_id": "C1",
                            "type": "file_exists",
                            "description": "检查文件",
                            "parameters": {"path": "b.py"}
                        }
                    ]
                }
            ]
        }
        with self.assertRaises(ValueError):
            self.planner._parse_response(json.dumps(data))
    def test_valid_acceptance_criteria_parsed(self):
        response = self._build_response()
        tasks = self.planner._parse_response(response)
        self.assertEqual(len(tasks), 1)
        criterion = tasks[0]["acceptance_criteria"][0]
        self.assertEqual(criterion.criterion_id, "C1")
        self.assertEqual(criterion.type, "file_exists")
        self.assertEqual(criterion.parameters, {"path": "app.py"})
    def test_all_supported_criterion_types(self):
        cases = [
            ("file_exists", {"path": "app.py"}),
            ("directory_exists", {"path": "src"}),
            ("file_contains", {"path": "app.py", "text": "def main"}),
            ("python_syntax", {"path": "app.py"}),
            ("json_valid", {"path": "config.json"})
        ]
        for criterion_type, parameters in cases:
            with self.subTest(criterion_type=criterion_type):
                response = self._build_response(
                    parameters=parameters,
                    criterion_type=criterion_type
                )
                tasks = self.planner._parse_response(response)
                self.assertEqual(
                    tasks[0]["acceptance_criteria"][0].type,
                    criterion_type
                )
    def test_dependency_duplicate_rejected(self):
        data = {
            "tasks": [
                {
                    "task_id": "T1",
                    "description": "任务一",
                    "dependencies": ["T2", " T2 "],
                    "acceptance_criteria": [
                        {
                            "criterion_id": "C1",
                            "type": "file_exists",
                            "description": "检查文件",
                            "parameters": {"path": "a.py"}
                        }
                    ]
                },
                {
                    "task_id": "T2",
                    "description": "任务二",
                    "dependencies": [],
                    "acceptance_criteria": [
                        {
                            "criterion_id": "C1",
                            "type": "file_exists",
                            "description": "检查文件",
                            "parameters": {"path": "b.py"}
                        }
                    ]
                }
            ]
        }
        with self.assertRaises(ValueError):
            self.planner._parse_response(json.dumps(data))
    def test_invalid_parameter_name_type_rejected(self):
        response = self._build_response(parameters={})
        data = json.loads(response)
        data["tasks"][0]["acceptance_criteria"][0]["parameters"] = {
            123: "app.py"
        }
        with self.assertRaises(ValueError):
            self.planner._parse_response(json.dumps(data))
    def test_parameter_name_with_internal_space_rejected(self):
        response = self._build_response(parameters={"pa th": "app.py"})
        with self.assertRaises(ValueError):
            self.planner._parse_response(response)


if __name__ == "__main__":
    unittest.main()
