
import json
import unittest

from nanobot.planner.planner import Planner

class TestAcceptanceCriteria(unittest.TestCase):

    def setUp(self):
        # _parse_response 不依赖 runtime，因此这里不需要真实 LLM
        self.planner = Planner(runtime=None)

    def make_criterion(
        self,
        criterion_id="C1",
        criterion_type="file_exists",
        parameters=None,
    ):
        if parameters is None:
            parameters = {"path": "app.py"}

        return {
            "criterion_id": criterion_id,
            "type": criterion_type,
            "description": "检查验收条件",
            "parameters": parameters,
        }

    def make_task(self, criteria=None, task_id="T1"):
        if criteria is None:
            criteria = [self.make_criterion()]

        return {
            "task_id": task_id,
            "description": "实现任务",
            "dependencies": [],
            "acceptance_criteria": criteria,
        }

    def make_response(self, tasks):
        return json.dumps({"tasks": tasks}, ensure_ascii=False)

    def parse(self, tasks):
        return self.planner._parse_response(
            self.make_response(tasks)
        )

    # 1. 五种支持的验收类型都应该通过
    def test_all_supported_criterion_types(self):
        criteria = [
            self.make_criterion(
                "C1", "file_exists", {"path": "app.py"}
            ),
            self.make_criterion(
                "C2", "directory_exists", {"path": "src"}
            ),
            self.make_criterion(
                "C3", "file_contains",
                {"path": "app.py", "text": "def main"}
            ),
            self.make_criterion(
                "C4", "python_syntax", {"path": "app.py"}
            ),
            self.make_criterion(
                "C5", "json_valid", {"path": "config.json"}
            ),
        ]

        tasks = self.parse([self.make_task(criteria)])

        self.assertEqual(len(tasks), 1)
        self.assertEqual(len(tasks[0]["acceptance_criteria"]), 5)

    # 2. 允许空验收标准列表
    def test_empty_acceptance_criteria(self):
        tasks = self.parse([
            self.make_task(criteria=[])
        ])

        self.assertEqual(
            tasks[0]["acceptance_criteria"], []
        )

    # 3. 不支持的验收类型应该拒绝
    def test_unsupported_criterion_type(self):
        criterion = self.make_criterion(
            criterion_type="unknown_type"
        )

        with self.assertRaises(ValueError):
            self.parse([self.make_task([criterion])])

    # 4. 缺少必需参数应该拒绝
    def test_missing_required_parameter(self):
        criterion = self.make_criterion(
            criterion_type="file_contains",
            parameters={"path": "app.py"},
        )

        with self.assertRaises(ValueError):
            self.parse([self.make_task([criterion])])

    # 5. 存在多余参数应该拒绝
    def test_extra_parameter(self):
        criterion = self.make_criterion(
            criterion_type="file_exists",
            parameters={
                "path": "app.py",
                "extra": "unexpected",
            },
        )

        with self.assertRaises(ValueError):
            self.parse([self.make_task([criterion])])

    # 6. 参数名为空应该拒绝
    def test_empty_parameter_name(self):
        criterion = self.make_criterion(
            parameters={"   ": "app.py"}
        )

        with self.assertRaises(ValueError):
            self.parse([self.make_task([criterion])])

    # 7. 参数值不限制为字符串
    def test_parameter_value_can_be_non_string(self):
        criterion = self.make_criterion(
            parameters={"path": 123}
        )

        tasks = self.parse([self.make_task([criterion])])

        self.assertEqual(
            tasks[0]["acceptance_criteria"][0].parameters["path"],
            123,
        )

    # 8. 同一个任务中 criterion_id 重复应该拒绝
    def test_duplicate_criterion_id_in_same_task(self):
        criteria = [
            self.make_criterion("C1"),
            self.make_criterion("C1", "python_syntax"),
        ]

        with self.assertRaises(ValueError):
            self.parse([self.make_task(criteria)])

    # 9. 不同任务可以使用相同的 criterion_id
    def test_same_criterion_id_in_different_tasks(self):
        tasks = self.parse([
            self.make_task(
                [self.make_criterion("C1")],
                task_id="T1",
            ),
            self.make_task(
                [self.make_criterion("C1")],
                task_id="T2",
            ),
        ])

        self.assertEqual(len(tasks), 2)

    # 10. criterion_id 为空应该拒绝
    def test_empty_criterion_id(self):
        criterion = self.make_criterion(
            criterion_id="   "
        )

        with self.assertRaises(ValueError):
            self.parse([self.make_task([criterion])])

    # 11. criterion_id 前后空格应该被清理
    def test_criterion_id_is_stripped(self):
        criterion = self.make_criterion(
            criterion_id=" C1 "
        )

        tasks = self.parse([self.make_task([criterion])])

        self.assertEqual(
            tasks[0]["acceptance_criteria"][0].criterion_id,
            "C1",
        )

    # 12. 缺少 acceptance_criteria 应该拒绝
    def test_missing_acceptance_criteria(self):
        task = {
            "task_id": "T1",
            "description": "实现任务",
            "dependencies": [],
        }

        with self.assertRaises(ValueError):
            self.parse([task])

    # 13. acceptance_criteria 不是列表应该拒绝
    def test_acceptance_criteria_not_list(self):
        task = self.make_task()
        task["acceptance_criteria"] = "C1"

        with self.assertRaises(ValueError):
            self.parse([task])

    # 14. 参数不是字典应该拒绝
    def test_parameters_not_dict(self):
        criterion = self.make_criterion()
        criterion["parameters"] = ["app.py"]

        with self.assertRaises(ValueError):
            self.parse([self.make_task([criterion])])


if __name__ == "__main__":
    unittest.main()
