from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import  StringSchema, tool_parameters_schema
import asyncio
import json

@tool_parameters(
    tool_parameters_schema(
        action=StringSchema(
            "需要执行的操作",
            enum=["list", "stats", "info", "delete", "clear","confirm_delete","confirm_clear"]
        ),
        filename=StringSchema(
            "需要查询的文件名称"
        ),
        source=StringSchema(
            "需要删除的文件的绝对路径"
        ),
        required=["action"]
    )
)

class KnowledgeBaseTool(Tool):
    @property
    def name(self)->str:
        return "knowledge_base"

    @property
    def description(self) -> str:
        return """知识库管理工具，通过 action 分发：
    - list：列出文档（filename、source、file_type、chunk_count）
    - stats：统计文档数、chunk 数、文件类型
    - info：按 filename 查详情
    - delete：预览待删文档，需 source
    - confirm_delete：确认删除，需 source
    - clear：预览清空影响
    - confirm_clear：确认清空
    规则：source 须来自 list；删除走 delete→用户确认→confirm_delete；清空走 clear→用户二次确认→confirm_clear；空库不删；未确认不得调用 confirm_*。""" 

    @property
    def read_only(self) -> bool:
        return False

    async def execute( self,action: str,filename: str = "",source: str = ""):
        from nanobot.rag_backend.rag.knowledge_base import list_documents,get_knowledge_base_stats,get_document_info,delete_document, clear_knowledge_base
        if action=="list":
            result=await asyncio.to_thread(list_documents)
        elif action=="stats":
            result=await asyncio.to_thread(get_knowledge_base_stats)
        elif action=="info":
            if not filename.strip():
                result="请提供文件名称"
            else:
                result=await asyncio.to_thread(get_document_info,filename)
        elif action=="delete":
            if not source.strip():
                result="请提供文件路径"
            else:
                result = None
                docs=await asyncio.to_thread(list_documents)
                for d in docs:
                    if d["source"]==source:
                        result=d
                        break
                if result: 
                    result["status"]="pending_confirmation"
                else:
                    result="未找到相关文件"
        elif action=="clear":
           result=await asyncio.to_thread(get_knowledge_base_stats)
           if result["total_chunks"]==0:
               result="知识库为空"
           else:
                result["status"]="pending_confirmation"
        elif action=="confirm_delete":
            if not source.strip():
                result="请提供文件路径"
            else:
                result=await asyncio.to_thread(delete_document,source)
                result["status"]="deleted" if result["success"] else "failed"
        elif action=="confirm_clear":
            result=await asyncio.to_thread(clear_knowledge_base)
            result["status"]="cleared" if result["success"] else "failed"
        else:
            result="不支持的操作"
        return json.dumps(result, ensure_ascii=False, indent=2)