from datetime import datetime
import asyncio
from .graph import TaskGraph
from .models import Task,TaskStatus

class DAGExecutor:
    def __init__(
        self,
        graph: TaskGraph,
        execute_task,
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

        self.graph = graph
        self.execute_task = execute_task
        self.max_concurrency = max_concurrency
        self.running_tasks = {}

    async def _execute_one(self,task):
        task.error = None
        try:
            task.status = TaskStatus.RUNNING
            if task.started_at is None:
                task.started_at = datetime.now()
            result=await self.execute_task(task)
        except asyncio.CancelledError:
            task.status=TaskStatus.CANCELLED
            task.completed_at=datetime.now()
            raise 
        except Exception as err:
            task.status=TaskStatus.FAILED
            task.completed_at=datetime.now()
            task.error=str(err)
            raise 
        task.status=TaskStatus.COMPLETED
        task.result = result
        task.completed_at = datetime.now()
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
                try:
                    async_task = asyncio.create_task(self._execute_one(t))
                    self.running_tasks[t.task_id] = async_task
                except Exception as err:
                    if t.retry_count<t.max_retries:
                        t.retry_count+=1
                        t.status = TaskStatus.PENDING
                        t.error=str(err)
                        t.completed_at = None
                    else:
                        t.status = TaskStatus.FAILED
                        t.error = str(err)
                        t.completed_at = datetime.now()
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
                            task.error = str(err)
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
        task.status=TaskStatus.BLOCKED