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
    if chunk_overlap>=0 and chunk_size>0 and chunk_size>chunk_overlap:
        x=chunk_size-chunk_overlap
    else:
        raise ValueError("参数不合法")
    for l in documents:
        content=l.page_content
        meta=l.metadata
        index=0
        chunk_index=0
        original_length=len(content)
        while index<original_length:
            m=meta.copy()
            m["index"]=chunk_index
            doc=Document(content[index:index+chunk_size],m) 
            rlist.append(doc)
            index+=x
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

def get_collection():
    p=os.path.join(current_dir, "..", "chromadb")
    chr_client = chromadb.PersistentClient(path=p)
    collection = chr_client.get_or_create_collection(name="knowledge_base")
    return collection

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


def ingest_documents(spl_list, emb_list):
    if not spl_list:
        return
    meta_list=[]
    raw_context_list=[]
    for a in spl_list:
        meta_list.append(a.metadata)
        raw_context_list.append(a.page_content)
    collection=get_collection()
    ids=[]
    for i in range(len(spl_list)):
        id=meta_list[i]["source"]+"_"+str(meta_list[i]["index"])
        ids.append(id)
    collection.upsert(ids=ids,documents=raw_context_list,embeddings=emb_list,metadatas=meta_list)
   
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
    
