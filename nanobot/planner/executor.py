from datetime import datetime
import asyncio
from pathlib import Path
from .validation import TaskResultValidator
from .graph import TaskGraph
from .models import Task,TaskStatus,ValidationStatus

class DAGExecutor:
    def __init__(
        self,
        graph: TaskGraph,
        execute_task,
        validator: TaskResultValidator,
        workspace_root: Path,
        max_concurrency: int = 3,
    ):
        if not callable(execute_task):
            raise ValueError("执行函数必须可调用")
        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency <= 0
        ):
            raise ValueError("最大并发数必须是正整数")
        if not isinstance(validator, TaskResultValidator):
            raise ValueError("验收器类型不正确")
        if not isinstance(workspace_root, Path):
            raise ValueError("工作空间根目录必须是 Path 类型")
        if not workspace_root.exists() or not workspace_root.is_dir():
            raise ValueError("工作空间根目录不存在或不是目录")
        self.graph = graph
        self.execute_task = execute_task
        self.max_concurrency = max_concurrency
        self.running_tasks = {}
        self.workspace_root=workspace_root.resolve()
        self.validator=validator

    async def _execute_one(self,task):
        try:
            task.status = TaskStatus.RUNNING
            task.validation_status = ValidationStatus.NOT_CHECKED
            task.validation_result = None
            if task.started_at is None:
                task.started_at = datetime.now()
            result=await self.execute_task(task)
        except asyncio.CancelledError:
            task.status=TaskStatus.CANCELLED
            task.validation_status = ValidationStatus.NOT_CHECKED
            task.validation_result = None
            task.completed_at=datetime.now()
            raise 
        except Exception as err:
            task.status=TaskStatus.FAILED
            task.completed_at=datetime.now()
            task.error=str(err)
            task.validation_status = ValidationStatus.NOT_CHECKED
            task.validation_result = None
            raise 
        task.result = result
        task.error = None
        task.validation_status = ValidationStatus.NOT_CHECKED
        task.validation_result = None
        validation_result = self.validator.validate(
            task.acceptance_criteria,
            self.workspace_root
        )
        task.validation_result = validation_result
        if not validation_result.passed:
            task.validation_status = ValidationStatus.FAILED
            task.error = "任务验收失败"
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            raise RuntimeError("任务验收失败")
        task.validation_status = ValidationStatus.PASSED
        task.status=TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.error = None
        return result

    async def execute(self):
        while(True):
            to_cancel_task = self.graph.mark_blocked_tasks()
            for t in to_cancel_task:
                await self.cancel_task(t)
                self.running_tasks.pop(t.task_id, None)
            over,ready_tasks=self.graph.get_ready_tasks()
            if over:
                break
            i=0
            available=self.max_concurrency-len(self.running_tasks)
            available = min(available, len(ready_tasks))
            while(i<available):
                t=ready_tasks[i]
                async_task = asyncio.create_task(self._execute_one(t))
                self.running_tasks[t.task_id] = async_task
                i+=1
            if not ready_tasks and not self.running_tasks and not over:
                raise RuntimeError("DAG 调度停滞：存在未完成任务，但没有可执行或运行中的任务")
            if self.running_tasks:
                done, pending = await asyncio.wait(
                    self.running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for async_task in done:
                    task_id = None
                    for key, running_task in list(self.running_tasks.items()):
                        if running_task is async_task:
                            task_id = key
                            del self.running_tasks[key]
                            break
                    if task_id is None:
                        continue
                    task = self.graph.get_task(task_id)
                    try:
                        async_task.result()
                    except asyncio.CancelledError:
                        pass
                    except Exception as err:
                        if task.retry_count < task.max_retries:
                            task.retry_count += 1
                            task.status = TaskStatus.PENDING
                            task.completed_at=None
                        else:
                            task.status = TaskStatus.FAILED                
                            task.completed_at = datetime.now()
        
                
                    
    async def cancel_task(self, task):
        task_id=task.task_id
        async_task = self.running_tasks.get(task_id)
        if async_task is not None:
            async_task.cancel()
            try:
                await async_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        task.status=TaskStatus.BLOCKED