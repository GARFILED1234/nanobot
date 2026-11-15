import os

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
        return "用于查询本地 RAG 知识库中的文档内容。只要用户的问题涉及本地文件、已导入的文档、项目资料、知识库中的信息，或者询问可能记录在本地文档中的具体事实、编号、名称、密码提示、测试暗号等，都应该优先调用此工具检索，不要仅凭模型自身知识回答。"

    @property
    def read_only(self)->bool:
        return True

    async def execute(self, query: str) -> str:
        from nanobot.rag_backend.rag.rag_chain import rag_answer
        from nanobot.rag_backend.rag.sync import sync_documents
        current = os.path.dirname(os.path.abspath(__file__))
        directory_path = os.path.abspath(
            os.path.join(current, "..", "..", "rag_backend", "documents")
        )
        await asyncio.to_thread(sync_documents,directory_path)
        return await asyncio.to_thread(rag_answer, query)