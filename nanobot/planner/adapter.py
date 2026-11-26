
from typing import TYPE_CHECKING

from nanobot.agent.tools.base import ToolResult
from nanobot.agent.tools.context import current_request_context
from nanobot.security.workspace_access import current_workspace_scope
from nanobot.session.manager import SessionPolicy

from .models import Task,ValidationStatus
from .graph import TaskGraph

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class SubagentTaskExecutor:
    def __init__(self, manager: "SubagentManager",graph:TaskGraph,disabled_tools: list[str] | None = None):
        self.graph=graph
        self.manager = manager
        self.disabled_tools = frozenset(disabled_tools or ())

    async def __call__(self, task: Task) -> str:
        request_ctx = current_request_context()
        if request_ctx is None or request_ctx.runtime is None:
            raise RuntimeError("缺少当前请求的模型运行时")
        origin_channel = request_ctx.channel
        origin_chat_id = request_ctx.chat_id
        session_key = request_ctx.session_key or (
            f"{origin_channel}:{origin_chat_id}"
        )
        if not session_key:
            raise RuntimeError("缺少当前会话标识")
        pre_result=f"用户原始需求：{request_ctx.original_user_text}"
        if self.disabled_tools:
            pre_result+=f"\n禁止使用的工具列表：{self.disabled_tools}"
        for t_id in task.dependencies:
            tmp_str=f"\n前置任务：{t_id}"
            t=self.graph.get_task(t_id)
            tmp_str+=f"\n任务描述：{t.description}"
            tmp_str+=f"\n执行结果：{t.result}"
            pre_result+=tmp_str
        if task.dependencies:
            pre_result+="""
- 前面列出的前置任务均已完成。
- 前置任务的执行结果仅作为当前任务的参考上下文。
- 应结合这些结果完成当前任务，不要无故重复执行前置任务。
- 如果前置结果不足以完成当前任务，应明确说明缺少什么信息。
"""
        if task.validation_status==ValidationStatus.FAILED and task.retry_count>0 and task.validation_result is not None:
            if task.result is not None:
                pre_result+=f"\n上一轮 Subagent 执行结果：{str(task.result)}"
            failed=[c for c in task.validation_result.criterion_results if c.passed is False]
            if failed:
                failed_text = "\n".join(
                    f"验收标准 ID：{c.criterion_id}\n失败原因：\n" +
                    "\n".join(f"- {item}" for item in c.feedback)
                    for c in failed
                )
            else:
                failed_text = "整体验收状态为失败，但没有找到具体未通过的验收标准。请检查当前工作区和上一轮执行结果，重新分析任务未通过的原因，不要自行修改或删除验收标准。"
            pre_result += f"""\n上一轮未通过的验收标准：{failed_text}
验收失败后的修复要求：
- 当前任务此前已经执行过，但未通过验收。你需要在现有实现的基础上进行修复，而不是将任务当作全新任务重新执行。
- 仔细阅读上一轮的执行结果、验收反馈和具体未通过的验收标准，分析每个问题的根本原因。
- 检查当前工作区中的已有文件和代码，确认实际实现情况，不要仅凭上一轮的执行总结判断。
- 优先针对验收失败的问题进行最小范围修改，避免无关修改、重复创建文件或推倒重写已有实现。
- 如果多个验收标准失败，应逐项分析并修复，不能只处理其中一项。
- 不要擅自降低、忽略或删除原有验收标准来使任务通过。
- 修复完成后，检查修改结果，确保满足当前任务的全部验收标准。
- 不要伪造执行结果或声称已经通过验收；如果无法修复，必须明确说明原因以及尚未解决的问题。
"""
        result = await self.manager.run_inline(
            task=f"{pre_result}\n当前任务：{task.description}",
            label=task.task_id,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
            session_key=session_key,
            origin_message_id=request_ctx.message_id,
            runtime=request_ctx.runtime,
            origin_turn_id=request_ctx.turn_id,
            workspace_scope=current_workspace_scope(),
            session_policy=SessionPolicy(
                persist=request_ctx.persist_session,
                log_content=request_ctx.log_content,
            ),
            disabled_tools=self.disabled_tools
        )
        if isinstance(result, ToolResult) and result.is_error:
            raise RuntimeError(str(result))
        return str(result)


