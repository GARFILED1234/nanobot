from .rag_chain import get_existing_files, load_documents,ingest_documents,get_collection,_prepare_documents

def _get_changed_documents(directory_path, force_rebuild=False):
    docs=load_documents(directory_path)
    file_dict=get_existing_files()
    todo_doc=[]
    for d in docs:
        m=d.metadata
        source=m.get("source")
        if not d.page_content.strip() and source not in file_dict:
            continue
        newhash=m.get("content_hash")
        oldhash=file_dict.get(source)
        if newhash!=oldhash or force_rebuild:
            todo_doc.append(d)
    return todo_doc,docs

def sync_documents(directory_path,force_rebuild=False):
    todo_doc,docs=_get_changed_documents(directory_path,force_rebuild)
    col=get_collection()
    old_m=col.get().get("metadatas")
    source_list=[om.get("source") for om in old_m if om is not None and om.get("source") is not None]
    source_list = list(set(source_list))
    todo_sources=[t.metadata["source"] for t in todo_doc]
    to_add_sources=[t for t in todo_sources if t not in source_list]
    to_update_sources=[s for s in todo_sources if s in source_list]
    new_source_list=[d.metadata.get("source") for d in docs]
    to_delete_sources=[olds for olds in source_list if olds not in new_source_list]
    if not todo_sources and not to_delete_sources:
        return
    backup_sources=to_delete_sources+to_update_sources
    backup_list=[]
    for s in backup_sources:
        backup_list.extend(_get_source_records(col,s))
    spl_list,emb_list=_prepare_documents(todo_doc)
    try:
        for s in backup_sources:
            col.delete(where={"source": s})
        ingest_documents(spl_list,emb_list,col)
    except Exception:
        try:
            ids=[]
            docs=[]
            embs=[]
            metas=[]
            for s in backup_sources + to_add_sources:
                col.delete(where={"source": s})
            if backup_list:
                for t in backup_list:
                    ids.append(t["id"])
                    docs.append(t["content"])
                    embs.append(t["emb"].tolist())
                    metas.append(t["metadata"])
                col.upsert(
                        ids=ids,
                        documents=docs,
                        embeddings=embs,
                        metadatas=metas
                    )
        except Exception as rollback_error:
            raise RuntimeError(
                f"增量同步失败，并且回滚也失败：{rollback_error}"
            ) from rollback_error
        raise

def _get_source_records(collection, source):
    result = collection.get(
        where={"source": source},
        include=["documents", "embeddings", "metadatas"]
    )
    docs=result["documents"]
    emb_list=result["embeddings"]
    ids=result["ids"]
    metas=result["metadatas"]
    r_list=[]
    for i in range(len(ids)):
        m=metas[i]
        if m is None:
            continue
        file_dict={}
        file_dict["metadata"]=m
        file_dict["id"]=ids[i]
        file_dict["content"]=docs[i]
        file_dict["emb"]=emb_list[i]
        r_list.append(file_dict)
    return r_list







