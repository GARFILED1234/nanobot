from .models import Task,TaskStatus
from collections import deque

class TaskGraph:
    def __init__(self):
        self.tasks: dict[str, Task] = {}

    def add_task(self, task):
        task_id=task.task_id
        if task_id in self.tasks:
            raise ValueError("当前任务已存在")
        self.tasks[task_id]=task
            
    def get_task(self, task_id):
        self._is_task_exist(task_id)
        return self.tasks[task_id]

    def add_dependency(self,task_id, dependency_id):
        dependent=self.get_task(dependency_id)
        task=self.get_task(task_id)
        if dependency_id in task.dependencies or task_id in dependent.dependents:
            raise ValueError("重复依赖")
        if task_id == dependency_id:
            raise ValueError("任务不能依赖自己")
        task.dependencies.append(dependency_id)
        dependent.dependents.append(task_id)
        if self._has_cycle():
            task.dependencies.remove(dependency_id)
            dependent.dependents.remove(task_id)
            raise ValueError("存在依赖循环,任务添加失败")

    def _is_task_exist(self,task_id):
        if task_id is None or task_id not in self.tasks:
            raise ValueError("当前任务不存在")

    def get_ready_tasks(self):
        ready_list=[]
        over=True
        for t in self.tasks.values():
            if t.status in (TaskStatus.PENDING,TaskStatus.READY,TaskStatus.RUNNING):
                over=False
            if t.status not in (TaskStatus.PENDING, TaskStatus.READY):
                continue
            if all(self.get_task(dep_id).status == TaskStatus.COMPLETED for dep_id in t.dependencies):
                t.status=TaskStatus.READY
                ready_list.append(t)
            else:
                t.status=TaskStatus.PENDING
        return over,ready_list

    def mark_blocked_tasks(self):
        visited=set()
        to_cancel_task=[]
        for t in self.tasks.values():
            if t.task_id in visited:
                continue
            if t.status not in (
                TaskStatus.BLOCKED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ):
                continue
            visited.add(t.task_id)
            visiting=deque(t.dependents.copy())
            while visiting:
                d=visiting.popleft()
                if d in visited:
                    continue      
                d_task=self.get_task(d)
                if d_task.status==TaskStatus.RUNNING:
                    to_cancel_task.append(d_task)
                elif d_task.status in (TaskStatus.PENDING, TaskStatus.READY):
                    d_task.status = TaskStatus.BLOCKED
                visiting.extend(d_task.dependents)
                visited.add(d)
        return to_cancel_task
            

    def validate(self):
        is_checked=set()
        for t in self.tasks.values():
            deps=t.dependencies
            for d in deps:
                if d in is_checked:
                    continue
                self._is_task_exist(d)
                is_checked.add(d)
        if self._has_cycle():
            raise ValueError("存在依赖循环")

    def _has_cycle(self):
        visited = set()
        visiting = set()

        def dfs(task_id):
            if task_id in visiting:
                return True
            if task_id in visited:
                return False
            visiting.add(task_id)
            task=self.get_task(task_id)
            for dep in task.dependencies:
                if dfs(dep):
                    return True
            visiting.remove(task_id)
            visited.add(task_id)
            return False

        for task_id in self.tasks:
            if dfs(task_id):
                return True
        return False



    

