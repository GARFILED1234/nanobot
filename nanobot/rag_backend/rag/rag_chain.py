import re
import json
import chromadb
import os
from dotenv import load_dotenv
from nanobot.utils.document import extract_text
from zai import ZhipuAiClient
import hashlib
import jieba
from typing import Dict, Any
from rank_bm25 import BM25Okapi
from fractions import Fraction
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, "..", ".env"))
client = ZhipuAiClient(api_key=os.getenv("ZHIPUAI_API_KEY"))
ext_set = {".txt", ".md", ".pdf", ".docx",
               ".xlsx", ".pptx", ".csv", ".json", ".xml", ".html", ".htm", ".log", ".yaml", ".yml", ".toml", ".ini", ".cfg"}

def _load_document(file_path):
    _, extension = os.path.splitext(file_path)
    if extension.lower() not in ext_set:
        return None
    content = extract_text(file_path)
    if content is None or not content.strip():
        return None
    if content.startswith("[error:"):
        raise OSError(content)
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    d={}
    d["source"]=file_path
    d["file_type"]=extension.lstrip(".").lower()
    d["filename"]=os.path.basename(file_path)
    d["content_hash"] = content_hash
    return Document(content, d)

def load_documents(directory_path):
    dp=os.listdir(directory_path)
    my_list=[]
    for d in dp:
        file_path = os.path.join(directory_path, d)
        if not os.path.isfile(file_path):
            continue
        doc=_load_document(file_path)
        if doc is not None:
            my_list.append(doc)
    return my_list

def embed_texts(texts: list[str]):
    response= client.embeddings.create(
        model="embedding-3",
        input=texts
    )
    rlist=[]
    for a in response.data:
        rlist.append(a.embedding)
    return rlist

def _retrieve_documents(query,n_results:int=2):
    e_query=embed_texts([query])
    collection = get_collection()
    r=collection.query(query_embeddings=e_query,n_results=n_results)

    documents = r["documents"][0]
    metadatas = r["metadatas"][0]
    ids = r["ids"][0]
    return [
        Document(
            page_content=content,
            metadata={**(metadata or {}), "id": id},
        )
        for content, metadata, id in zip(documents, metadatas, ids)
    ]

def split_documents(documents,chunk_size:int =100,chunk_overlap:int=20):
    rlist=[]
    if chunk_overlap<0 or chunk_size<=0 or chunk_size<=chunk_overlap:
        raise ValueError("参数不合法")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ""]
    )
    for l in documents:
        content=l.page_content
        meta=l.metadata
        chunk_index=0
        file_type = meta.get("file_type", "").lower().lstrip(".")
        if file_type=="md":
            contents=_split_markdown_sections(content)
            for d in contents:
                docs=splitter.split_text(d["content"])
                for do in docs:
                    m=meta.copy()
                    m["index"]=chunk_index
                    m["heading_path"]=d["heading_path"]
                    doc=Document(do,m)
                    rlist.append(doc)
                    chunk_index+=1
        else:
            docs=splitter.split_text(content)
            for d in docs:
                m=meta.copy()
                m["index"]=chunk_index
                doc=Document(d,m)
                rlist.append(doc)
                chunk_index+=1
    return rlist

def _build_rag_prompt(query,n_results:int=2):
    candidate_k = 10
    docs = _hybrid_retrieve_documents(query, candidate_k)
    docs=_rerank_documents(query,docs,n_results)
    context=""
    for d in docs:
        filename = d.metadata.get("filename", "未知文件")
        context=f"{context}\n内容：{d.page_content}\n来源：{filename}"
    prompt={
        "context":context,
        "query":query
    }
    return prompt

def get_chroma_client():
    p=os.path.join(current_dir, "..", "chromadb")
    return chromadb.PersistentClient(path=p)

def get_collection():
    chr_client = get_chroma_client()
    collection = chr_client.get_or_create_collection(name="knowledge_base")
    return collection

def get_rebuild_collection():
    chr_client=get_chroma_client()
    return chr_client.get_or_create_collection(name="knowledge_base_rebuild")

def rebuild_collection(directory_path, chunk_size=100, chunk_overlap=20):
    docs = load_documents(directory_path)
    if not docs:
        raise ValueError("没有加载到有效文档，无法重建知识库")
    spl, emb = _prepare_documents(docs, chunk_size, chunk_overlap)
    if not spl:
        raise ValueError("没有生成有效分块，无法重建知识库")
    col = get_rebuild_collection()
    if col.count() > 0:
        col.delete(where={})
    ingest_documents(
        spl_list=spl,
        emb_list=emb,
        collection=col
    )
    _switch_rebuild_collection()
    client = get_chroma_client()
    client.delete_collection(name="knowledge_base_backup")

def _switch_rebuild_collection():
    client = get_chroma_client()
    names = [collection.name for collection in client.list_collections()]
    if "knowledge_base" not in names:
        raise RuntimeError("正式知识库集合 knowledge_base 不存在")
    if "knowledge_base_rebuild" not in names:
        raise RuntimeError("重建集合 knowledge_base_rebuild 不存在")
    if "knowledge_base_backup" in names:
        raise RuntimeError(
            "备份集合 knowledge_base_backup 已存在，"
            "为避免覆盖旧备份，取消本次切换"
        )
    old_col = client.get_collection(name="knowledge_base")
    rebuild_col = client.get_collection(name="knowledge_base_rebuild")
    # 第一步：备份旧集合
    old_col.modify(name="knowledge_base_backup")
    # 第二步：将重建集合切换为正式集合
    try:
        rebuild_col.modify(name="knowledge_base")
    except Exception as switch_error:
        # 第二步失败，尝试恢复旧集合
        try:
            backup_col = client.get_collection(
                name="knowledge_base_backup"
            )
            backup_col.modify(name="knowledge_base")
        except Exception as rollback_error:
            raise RuntimeError(
                f"集合切换失败，且旧集合恢复失败：{rollback_error}"
            ) from rollback_error
        raise RuntimeError(
            f"集合切换失败，旧集合已恢复：{switch_error}"
        ) from switch_error

def rag_answer(query,n_results:int=2):
    prompt=_build_rag_prompt(query,n_results)
    messages=[
        {
            "role":"system",
            "content":"你是一个知识回答助手"
        },{
            "role":"user",
            "content":json.dumps(prompt, ensure_ascii=False)
        }
    ]
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content

def _prepare_documents(documents,chunk_size=100, chunk_overlap=20):
    spl_list=split_documents(documents,chunk_size,chunk_overlap)
    if not spl_list:
        return [],[]
    raw_context_list=[]
    for a in spl_list:
        raw_context_list.append(a.page_content)
    emb_list=embed_texts(raw_context_list)
    return spl_list,emb_list


def ingest_documents(spl_list, emb_list,collection):
    if len(spl_list) != len(emb_list):
        raise ValueError("分块数量与 Embedding 数量不一致")
    if not spl_list:
        return
    meta_list=[]
    raw_context_list=[]
    for a in spl_list:
        meta_list.append(a.metadata)
        raw_context_list.append(a.page_content)
    ids=[]
    for i in range(len(spl_list)):
        id=meta_list[i]["source"]+"_"+str(meta_list[i]["index"])
        ids.append(id)
    collection.upsert(ids=ids,documents=raw_context_list,embeddings=emb_list,metadatas=meta_list)
    new_c=len(collection.get(ids=ids).get("ids"))
    if new_c!=len(spl_list):
        raise RuntimeError("向量写入数量不一致")

def get_existing_files():
    col=get_collection()
    met=col.get(include=["metadatas"])["metadatas"]
    file_dict={}
    for m in met:
        if m is None:
            continue
        source=m.get("source")
        con_hash=m.get("content_hash")
        if source is None or con_hash is None:
            continue
        file_dict[source]=con_hash
    return file_dict

def _bm25_retrieve_documents(query, n_results: int = 2):
    collection = get_collection()
    result = collection.get(
        include=["documents", "metadatas"]
    )
    if not result["documents"]:
        return []
    tokenized_query=list(jieba.cut(query))
    docs=result["documents"]
    tokenized_docs=[]
    for d in docs:
        tokenized_docs.append(list(jieba.cut(d)))
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)
    order_list=sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )
    rdocs=[]
    for i in order_list[:n_results]:
        content = result["documents"][i]
        metadata = (result["metadatas"][i] or {}).copy()
        chunk_id = result["ids"][i]
        metadata["id"] = chunk_id
        doc=Document(page_content=content,metadata=metadata)
        rdocs.append(doc)
    return rdocs

def _rrf_fusion(vector_docs, bm25_docs, n_results=2):
    score_dict={}
    for i in range(len(vector_docs)):
        vd=vector_docs[i]
        met=vd.metadata
        id=met.get("id")
        score_list={}
        score_list["score"] = score_dict.get(id, 0) + Fraction(1, i+61)
        score_list["doc"]=vd
        score_dict[id]=score_list
    for j in range(len(bm25_docs)):
        bd=bm25_docs[j]
        met=bd.metadata
        id=met.get("id")
        ds=Fraction(1, j+61)
        score_list={}
        vs=0
        if id in score_dict:
            score_list=score_dict[id]
            vs=score_list["score"]
        else:
            score_list["doc"]=bd
        score_list["score"]=ds+vs
        score_dict[id]=score_list
    docs = [
        v["doc"]
        for k, v in sorted(score_dict.items(), key=lambda kv: kv[1]["score"], reverse=True)
    ]
    return docs[:n_results]

def _hybrid_retrieve_documents(query, n_results=2):
    k=10
    count=get_collection().count()
    if count == 0:
        return []
    k=min(k,count)
    vector_d=_retrieve_documents(query,k)
    bm25_d=_bm25_retrieve_documents(query,k)
    docs=_rrf_fusion(vector_docs=vector_d,bm25_docs=bm25_d,n_results=n_results)
    return docs

def _rerank_documents(query, docs, n_results=2):
    if docs is None or len(docs)==0:
        return []
    documents = [d.page_content for d in docs]
    response=client.post(
        "/rerank",
        cast_type=Dict[str, Any],
        body={
            "model": "rerank",
            "query": query,
            "documents": documents,
            "top_n": n_results
        }
    )
    r_docs=[]
    for r in response["results"]:
        i=r["index"]
        if 0 <= i < len(docs):
            r_docs.append(docs[i])
    return r_docs

def _split_markdown_sections(content):
    heading_stack = []
    contents = []
    current_content = []
    current_heading_path = ""
    has_current_section = False
    # 代码围栏状态
    in_code_block = False
    fence_char = None
    fence_length = 0
    for line in content.splitlines():
        # 检查代码围栏
        fence_match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        # 当前处于代码块内部
        if in_code_block:
            current_content.append(line)
            close_match = re.match(r"^ {0,3}(`{3,}|~{3,})\s*$", line)
            if (close_match is not None
                    and close_match.group(1)[0] == fence_char
                    and len(close_match.group(1)) >= fence_length):
                in_code_block = False
                fence_char = None
                fence_length = 0
            continue
        # 遇到代码围栏开始
        if fence_match is not None:
            if not has_current_section:
                has_current_section = True
                current_heading_path = ""
            current_content.append(line)
            fence_char = fence_match.group(1)[0]
            fence_length = len(fence_match.group(1))
            in_code_block = True
            continue
        # 检查 Markdown 标题
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match is not None:
            # 先保存上一章节
            if has_current_section:
                contents.append({
                    "heading_path": current_heading_path,
                    "content": "\n".join(current_content)
                })
            # 更新标题栈
            level = len(match.group(1))
            title = match.group(2)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            # 生成新的标题路径
            current_heading_path = " > ".join(
                title for _, title in heading_stack
            )
            # 开始新章节
            current_content = []
            has_current_section = True
        else:
            # 没有标题的文档内容也要保留
            if not has_current_section:
                current_heading_path = ""
                has_current_section = True
            current_content.append(line)
    # 保存最后一个章节
    if has_current_section:
        contents.append({
            "heading_path": current_heading_path,
            "content": "\n".join(current_content)
        })
    return contents