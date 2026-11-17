from .rag_chain import get_collection

def list_documents():
    col=get_collection()
    metas=col.get(include=["metadatas"])["metadatas"]
    source_chunk={}
    for m in metas:
        source=m.get("source")
        if source in source_chunk:
            doc_dict=source_chunk[source]
            doc_dict["chunk_count"]+=1
        else:
            doc_dict={}
            doc_dict["filename"]=m.get("filename")
            doc_dict["source"]=source
            doc_dict["file_type"]=m.get("file_type")
            doc_dict["chunk_count"]=1
            source_chunk[source]=doc_dict
    return list(source_chunk.values())

def get_knowledge_base_stats():
    docs=list_documents()
    total_dict = {
        "total_documents": len(docs),
        "total_chunks": 0,
        "file_types": {}
    }
    for d in docs:
        total_dict["total_chunks"]+=d["chunk_count"]
        file_type=d["file_type"]
        file_types=total_dict["file_types"]
        if file_type in file_types:
            file_types[file_type]+=1
        else:
            file_types[file_type]=1
    return total_dict

def get_document_info(filename: str):
    d_list=[]
    docs=list_documents()
    for d in docs:
        if d["filename"]==filename:
            d_list.append(d)
    return d_list

def delete_document(source: str):
    col=get_collection()
    before_count = col.count()
    col.delete(where={"source": source})
    after_count = col.count()
    deleted = before_count > after_count
    r_dict={
        "success":deleted,
        "source":source,
        "message":"文档删除成功" if deleted else "删除失败"
    }
    return r_dict

def clear_knowledge_base():
    col=get_collection()
    n = col.count()
    r_dict={}
    r_dict["total_chunks"]=n
    if n==0:
        r_dict["message"]="无数据"
        r_dict["success"]=False
        return r_dict
    ids = col.get()["ids"]
    col.delete(ids=ids)
    r_dict["success"]=True
    r_dict["message"]="清空成功"
    return r_dict
    