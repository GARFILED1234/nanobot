
from typing import TYPE_CHECKING

from nanobot.agent.tools.base import ToolResult
from nanobot.agent.tools.context import current_request_context
from nanobot.security.workspace_access import current_workspace_scope
from nanobot.session.manager import SessionPolicy

from .models import Task
from .graph import TaskGraph

if TYPE_CHECKING:
    from nanobot.agent.subagent import SubagentManager


class SubagentTaskExecutor:
    def __init__(self, manager: "SubagentManager",graph:TaskGraph):
        self.graph=graph
        self.manager = manager

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
        pre_result=""
        for t_id in task.dependencies:
            tmp_str=f"\n前置任务：{t_id}"
            t=self.graph.get_task(t_id)
            tmp_str+=f"\n任务描述：{t.description}"
            tmp_str+=f"\n执行结果：{t.result}"
            pre_result+=tmp_str
        if pre_result:
            pre_result+="""
- 前面列出的前置任务均已完成。
- 前置任务的执行结果仅作为当前任务的参考上下文。
- 应结合这些结果完成当前任务，不要无故重复执行前置任务。
- 如果前置结果不足以完成当前任务，应明确说明缺少什么信息。
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
        )
        if isinstance(result, ToolResult) and result.is_error:
            raise RuntimeError(str(result))
        return str(result)


