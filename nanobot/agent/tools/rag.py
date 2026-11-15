from nanobot.agent.tools.base import Tool,  tool_parameters
from nanobot.agent.tools.schema import  StringSchema, tool_parameters_schema
import asyncio

@tool_parameters(
    tool_parameters_schema(
        query=StringSchema(
            "请输入问题"
        ),
        required=["query"]
    )
)

class RAGTool(Tool):
    @property
    def name(self)->str:
        return "rag"

    @property
    def description(self) -> str:
        return "用于查询和检索本地知识库。当用户询问关于特定文档或内部知识时，请调用此工具。"

    @property
    def read_only(self)->bool:
        return True

    async def execute(self, query: str) -> str:
        from nanobot.rag_backend.rag.rag_chain import rag_answer
        return await asyncio.to_thread(rag_answer, query)