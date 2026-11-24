from __future__ import annotations
from typing import TYPE_CHECKING,Any
from nanobot.agent.tools.base import Tool,  tool_parameters
from nanobot.agent.tools.schema import  StringSchema, tool_parameters_schema

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.agent.tools.context import ToolContext


@tool_parameters(
    tool_parameters_schema(
        goal=StringSchema(
            "输入完整的用户目标及要求，必须是字符串。所有任务拆分要求都写在 goal 中，不要额外传递 parameters、task1、task2 等参数。"
        ),
        required=["goal"],
        additional_properties=True
    )
)

class DAGPlannerTool(Tool):
    def __init__(self, manager: SubagentManager,max_concurrent_planner_tasks: int):
        self._manager = manager
        self.max_concurrent_planner_tasks=max_concurrent_planner_tasks

    @classmethod
    def create(cls, ctx: ToolContext) -> Tool:
        if ctx.subagent_manager is None:
            raise RuntimeError("DAGPlannerTool requires an initialized subagent manager")
        return cls(ctx.subagent_manager,ctx.max_concurrent_planner_tasks)

    @property
    def name(self) -> str:
        return "dag_planner"

    @property
    def description(self) -> str:
        return "用于将一个复杂问题拆分成可独立执行的子问题，分步解决问题"

    async def execute(self, **kwargs: Any) -> Any:
        print("[DAGPlanner Debug] execute entered", flush=True)
        print(f"[DAGPlanner Debug] kwargs: {kwargs!r}", flush=True)
        from ...planner.workflow import DAGWorkflow
        from nanobot.agent.tools.context import current_request_context
        import json
        ctx=current_request_context()
        print(f"[DAGPlanner Debug] request context: {ctx is not None}", flush=True)
        print(
            f"[DAGPlanner Debug] runtime available: {ctx is not None and ctx.runtime is not None}",
            flush=True,
        )
        if not ctx or not ctx.runtime:
            raise RuntimeError("缺少当前请求的模型运行时") 
        runtime=ctx.runtime
        dag_workflow=DAGWorkflow(self._manager,self.max_concurrent_planner_tasks)
        result=await dag_workflow.run(kwargs,runtime)
        return json.dumps(result,ensure_ascii=False, indent=2)
            
    @property
    def read_only(self)->bool:
        return False