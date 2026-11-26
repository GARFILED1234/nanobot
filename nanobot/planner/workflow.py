from .executor import DAGExecutor
from .planner import Planner
from .models import TaskStatus,ValidationStatus
from .adapter import SubagentTaskExecutor
from .validation.validator import TaskResultValidator
from .validation.registry import create_default_registry
from nanobot.security.workspace_access import current_workspace_scope

class DAGWorkflow:
    def __init__(self,manager, max_concurrency):
        self.manager=manager
        self.max_concurrency=max_concurrency

    async def run(self,paras, runtime):
        planner=Planner(runtime)
        disabled_tools=paras.get("disabled_tools")
        graph=await planner.plan(paras)
        workspace_scope = current_workspace_scope()
        workspace_root = workspace_scope.project_path if workspace_scope is not None else self.manager.workspace
        execute_task=SubagentTaskExecutor(self.manager,graph,disabled_tools)
        dag_executor=DAGExecutor(graph=graph,execute_task=execute_task,validator=TaskResultValidator(create_default_registry()),workspace_root=workspace_root,max_concurrency=self.max_concurrency)
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
            if t.validation_status==ValidationStatus.FAILED:
                tmp_dict["feedback"]=t.validation_result.feedback
            if t.status!=TaskStatus.COMPLETED:
                incomplete_tasks.append(tmp_dict)
                continue
            tmp_dict["result"]=t.result
            if not t.dependents:
                terminal_results.append(tmp_dict)
            else:
                process_results.append(tmp_dict)
        success = bool(graph.tasks) and all(
            t.status == TaskStatus.COMPLETED
            for t in graph.tasks.values()
        )
        result = {
            "success": success,
            "terminal_results": terminal_results,
            "process_results": process_results,
            "incomplete_tasks": incomplete_tasks
        }
        return result


