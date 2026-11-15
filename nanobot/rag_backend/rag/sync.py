from .rag_chain import get_existing_files, load_documents,ingest_documents,get_collection,_prepare_documents


def _get_changed_documents(directory_path):
    docs=load_documents(directory_path)
    file_dict=get_existing_files()
    todo_doc=[]
    for d in docs:
        m=d.metadata
        source=m.get("source")
        newhash=m.get("content_hash")
        oldhash=file_dict.get(source)
        if newhash!=oldhash:
            todo_doc.append(d)
    return todo_doc,docs

def sync_documents(directory_path):
    todo_doc,docs=_get_changed_documents(directory_path)
    col=get_collection()
    old_m=col.get().get("metadatas")
    source_list=[om.get("source") for om in old_m if om is not None and om.get("source") is not None]
    new_source_list=[d.metadata.get("source") for d in docs]
    for old_s in source_list:
        if old_s not in new_source_list:
            col.delete(where={"source": old_s})
    if not todo_doc:
        return
    spl_list,emb_list=_prepare_documents(todo_doc)
    for d in todo_doc:
        m=d.metadata
        source=m.get("source")
        col.delete(where={"source": source})
    ingest_documents(spl_list,emb_list)








