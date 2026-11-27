from __future__ import annotations
from typing import TYPE_CHECKING,Any
from nanobot.agent.tools.base import Tool,  tool_parameters
from nanobot.agent.tools.schema import  StringSchema, tool_parameters_schema,ArraySchema

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager
    from nanobot.agent.tools.context import ToolContext


@tool_parameters(
    tool_parameters_schema(
        goal=StringSchema(
            "输入完整的用户目标及要求，必须是字符串。所有任务拆分要求都写在 goal 中，不要额外传递 parameters、task1、task2 等参数。"
        ),
        disabled_tools=ArraySchema(
            items=StringSchema(),
            description=(
                "本轮 DAG 执行期间需要禁用的工具名称列表。"
                "只有用户明确要求本轮不使用某工具时才填写。"
                "例如：['web_search']。"
                "如果没有明确限制，则不传此参数。"
            ),
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
        return """用于将复杂目标拆分成具有依赖关系的子任务，并通过 Subagent 并行或按依赖顺序执行任务。
调用决策规则：
1. 当用户明确要求使用 DAG Planner、任务规划或多 Agent 协作时，必须调用本工具。
2. 当用户目标包含多个具有独立交付结果的子任务时，必须调用本工具。
3. 当一个任务需要多个阶段完成，且后续阶段依赖前面阶段的执行结果时，必须调用本工具。
4. 当用户要求开发涉及多个模块、多个文件或多个组件的完整功能时，必须调用本工具。
5. 当任务需要多个 Subagent 分别负责不同部分，并且需要统一协调执行顺序时，必须调用本工具。
6. 如果任务只涉及单个简单操作，例如读取一个文件、创建一个简单文件、修改一个配置项或执行一个简单命令，可以直接调用普通工具，不需要使用本工具。
7. 不要仅根据用户请求的文字长度判断任务复杂度，应根据实际交付目标、子任务数量和任务依赖关系进行判断。
8. 一旦判断需要使用 DAG Planner，应直接调用本工具，不要先通过普通工具执行其中一部分任务，再决定是否使用 DAG Planner。
9. 不要因为某个子任务可以通过一个普通工具完成，就跳过 DAG Planner。应根据整个用户目标判断是否需要规划。
参数使用规则：
1. 将用户的完整目标、具体要求和约束放入 goal，确保 Planner 能够获得完整的任务上下文。
2. 不要将任务拆分要求放在 goal 之外，也不要额外传递 parameters、task1、task2 等参数。
3. 只有用户明确要求本轮执行期间禁用某些工具时，才填写 disabled_tools。
4. disabled_tools 只影响本轮 DAG 执行，不代表永久禁用。用户没有明确提出工具限制时，不要填写该参数。
结果解释规则：
1. 必须检查返回结果中的 success 字段，判断整个 DAG 是否成功。
2. success=true 表示所有任务均已完成并通过验收，可以向用户报告目标已完成。
3. success=false 表示整个 DAG 未成功完成，不得向用户宣称整体目标已经完成。
4. terminal_results 保存已完成的终端任务结果，process_results 保存已完成的中间任务结果。它们只代表相应任务已经完成，不能单独作为整个 DAG 成功的依据。
5. 必须检查 incomplete_tasks，识别未完成的任务，并向用户如实说明任务状态及原因。
6. 区分不同类型的失败信息：
   - error：任务执行过程中发生异常。
   - feedback：任务未通过验收标准时产生的具体反馈。
   - fail_reason：任务因前置依赖失败而阻塞，或者任务被取消的原因。
7. 如果部分任务成功、部分任务失败，应分别说明已经完成的工作、未完成的工作及其具体原因。
8. 如果存在验收反馈，应将反馈作为后续处理的依据，不得忽略验收失败，也不得将未通过验收的任务描述为已完成。
9. 不要根据 Subagent 返回了结果、工具调用成功或部分任务完成，就推断整个用户目标已经实现。
10. 向用户汇报时，应以 DAG 返回的实际执行状态和验收结果为依据，不得编造完成情况或验收结论。
"""

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
        disabled_tools=kwargs.get("disabled_tools")
        if disabled_tools is not None:
            if not isinstance(disabled_tools,list):
                raise ValueError("禁用工具列表格式有误")
            if any(
                not isinstance(name, str) or not name.strip()
                for name in disabled_tools
            ):
                raise ValueError("存在不合法工具名")
        paras=kwargs.copy()
        if disabled_tools is not None:
            paras["disabled_tools"]=list(dict.fromkeys(name.strip() for name in disabled_tools))
        original_user_text=ctx.original_user_text
        if isinstance(original_user_text, str) and original_user_text.strip():
            paras["original_user_text"] = original_user_text.strip()
        dag_workflow=DAGWorkflow(self._manager,self.max_concurrent_planner_tasks)
        result=await dag_workflow.run(paras,runtime)
        return json.dumps(result,ensure_ascii=False, indent=2)
            
    @property
    def read_only(self)->bool:
        return False