
import unittest

from nanobot.planner.models import Task, TaskStatus
from nanobot.planner.graph import TaskGraph


class TestTaskGraph(unittest.TestCase):

    def setUp(self):
        self.graph = TaskGraph()

    def test_add_task(self):
        task = Task("T1", "任务一")
        self.graph.add_task(task)

        self.assertIs(self.graph.get_task("T1"), task)

    def test_add_duplicate_task(self):
        self.graph.add_task(Task("T1", "任务一"))

        with self.assertRaises(ValueError):
            self.graph.add_task(Task("T1", "重复任务"))

    def test_get_nonexistent_task(self):
        with self.assertRaises(ValueError):
            self.graph.get_task("T1")

    def test_add_dependency(self):
        self.graph.add_task(Task("T1", "任务一"))
        self.graph.add_task(Task("T2", "任务二"))

        self.graph.add_dependency("T2", "T1")

        self.assertEqual(
            self.graph.get_task("T2").dependencies,
            ["T1"]
        )

    def test_add_nonexistent_dependency(self):
        self.graph.add_task(Task("T1", "任务一"))

        with self.assertRaises(ValueError):
            self.graph.add_dependency("T1", "T2")

    def test_add_dependency_to_nonexistent_task(self):
        self.graph.add_task(Task("T1", "任务一"))

        with self.assertRaises(ValueError):
            self.graph.add_dependency("T2", "T1")

    def test_add_duplicate_dependency(self):
        self.graph.add_task(Task("T1", "任务一"))
        self.graph.add_task(Task("T2", "任务二"))

        self.graph.add_dependency("T2", "T1")

        with self.assertRaises(ValueError):
            self.graph.add_dependency("T2", "T1")

    def test_add_self_dependency(self):
        self.graph.add_task(Task("T1", "任务一"))

        with self.assertRaises(ValueError):
            self.graph.add_dependency("T1", "T1")

    def test_detect_indirect_cycle_and_rollback(self):
        self.graph.add_task(Task("T1", "任务一"))
        self.graph.add_task(Task("T2", "任务二"))
        self.graph.add_task(Task("T3", "任务三"))

        self.graph.add_dependency("T2", "T1")
        self.graph.add_dependency("T3", "T2")

        with self.assertRaises(ValueError):
            self.graph.add_dependency("T1", "T3")

        self.assertEqual(
            self.graph.get_task("T1").dependencies,
            []
        )
        self.assertEqual(
            self.graph.get_task("T2").dependencies,
            ["T1"]
        )
        self.assertEqual(
            self.graph.get_task("T3").dependencies,
            ["T2"]
        )

        self.assertEqual(len(self.graph.tasks), 3)

    def test_validate_normal_dag(self):
        self.graph.add_task(Task("T1", "任务一"))
        self.graph.add_task(Task("T2", "任务二"))
        self.graph.add_task(Task("T3", "任务三"))

        self.graph.add_dependency("T2", "T1")
        self.graph.add_dependency("T3", "T2")

        self.assertIsNone(self.graph.validate())

    def test_validate_missing_dependency(self):
        self.graph.add_task(Task("T1", "任务一"))
        self.graph.get_task("T1").dependencies.append("T2")

        with self.assertRaises(ValueError):
            self.graph.validate()

    def test_get_ready_tasks_without_dependencies(self):
        task = Task("T1", "任务一")
        self.graph.add_task(task)

        over, ready_tasks = self.graph.get_ready_tasks()

        self.assertEqual(ready_tasks, [task])
        self.assertFalse(over)
        self.assertEqual(task.status, TaskStatus.READY)

    def test_dependency_not_completed(self):
        task1 = Task("T1", "任务一")
        task2 = Task("T2", "任务二")

        self.graph.add_task(task1)
        self.graph.add_task(task2)
        self.graph.add_dependency("T2", "T1")

        over, ready_tasks = self.graph.get_ready_tasks()

        self.assertEqual(ready_tasks, [task1])
        self.assertFalse(over)
        self.assertEqual(task2.status, TaskStatus.PENDING)

    def test_task_becomes_ready_after_dependency_completed(self):
        task1 = Task("T1", "任务一")
        task2 = Task("T2", "任务二")

        self.graph.add_task(task1)
        self.graph.add_task(task2)
        self.graph.add_dependency("T2", "T1")

        self.graph.get_ready_tasks()

        task1.status = TaskStatus.COMPLETED

        over, ready_tasks = self.graph.get_ready_tasks()

        self.assertEqual(ready_tasks, [task2])
        self.assertFalse(over)
        self.assertEqual(task2.status, TaskStatus.READY)

    def test_ready_status_reverts_when_dependency_not_completed(self):
        task1 = Task("T1", "任务一")
        task2 = Task("T2", "任务二")

        self.graph.add_task(task1)
        self.graph.add_task(task2)
        self.graph.add_dependency("T2", "T1")

        task2.status = TaskStatus.READY

        over, ready_tasks = self.graph.get_ready_tasks()

        self.assertEqual(ready_tasks, [task1])
        self.assertFalse(over)
        self.assertEqual(task2.status, TaskStatus.PENDING)


if __name__ == "__main__":
    unittest.main()
