import json
import chromadb
import os
from dotenv import load_dotenv
from nanobot.utils.document import extract_text
from zai import ZhipuAiClient
import hashlib

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

    return [
        Document(content, metadata or {})
        for content, metadata in zip(documents, metadatas)
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
    docs=_retrieve_documents(query,n_results)
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


