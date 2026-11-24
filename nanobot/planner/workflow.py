from .executor import DAGExecutor
from .planner import Planner
from .models import TaskStatus
from .adapter import SubagentTaskExecutor

class DAGWorkflow:
    def __init__(self,manager, max_concurrency):
        self.manager=manager
        self.max_concurrency=max_concurrency

    async def run(self,paras, runtime):
        planner=Planner(runtime)
        graph=await planner.plan(paras)
        execute_task=SubagentTaskExecutor(self.manager,graph)
        dag_executor=DAGExecutor(graph,execute_task,self.max_concurrency)
        await dag_executor.execute()
        terminal_results = []
        process_results=[]
        incomplete_tasks = []
        for t in graph.tasks.values():
            tmp_dict={}
            tmp_dict["task_id"]=t.task_id
            tmp_dict["description"]=t.description
            tmp_dict["status"]=t.status.value
            if t.status==TaskStatus.FAILED:
                tmp_dict["error"]=t.error
            elif t.status==TaskStatus.BLOCKED:
                tmp_dict["fail_reason"]="因前置依赖失败而未执行"
            elif t.status==TaskStatus.CANCELLED:
                tmp_dict["fail_reason"]="任务已取消"
            if t.status!=TaskStatus.COMPLETED:
                incomplete_tasks.append(tmp_dict)
                continue
            tmp_dict["result"]=t.result
            if not t.dependents:
                terminal_results.append(tmp_dict)
            else:
                process_results.append(tmp_dict)
        result={
            "terminal_results":terminal_results,
            "process_results":process_results,
            "incomplete_tasks":incomplete_tasks
        }
        return result


