import json
import chromadb
import os
from dotenv import load_dotenv
from zai import ZhipuAiClient

class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(current_dir, "..", ".env"))
client = ZhipuAiClient(api_key=os.getenv("ZHIPUAI_API_KEY"))

def _load_text_file(file_path):
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    d={}
    d["source"]=file_path
    d["file_type"]="txt"
    d["filename"]=os.path.basename(file_path)
    return Document(content, d) 

def load_documents(directory_path):
    dp=os.listdir(directory_path)
    my_list=[]
    for d in dp:
        if(d.endswith(".txt")):
            my_list.append(_load_text_file(os.path.join(directory_path,d)))
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
    return r["documents"][0]

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
    context="\n".join(_retrieve_documents(query,n_results))
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

def ingest_documents(directory_path, chunk_size=100, chunk_overlap=20):
    raw_list=load_documents(directory_path)
    spl_list=split_documents(raw_list,chunk_size,chunk_overlap)
    raw_context_list=[]
    meta_list=[]
    for a in spl_list:
        raw_context_list.append(a.page_content)
        meta_list.append(a.metadata)
    emb_list=embed_texts(raw_context_list)
    collection=get_collection()
    ids=[]
    for i in range(len(spl_list)):
        id=meta_list[i]["filename"]+"_"+str(meta_list[i]["index"])
        ids.append(id)
    collection.upsert(ids=ids,documents=raw_context_list,embeddings=emb_list,metadatas=meta_list)
   




