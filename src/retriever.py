from typing import List, Dict
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from src.config import CHROMA_DIR, OPENAI_API_KEY, EMBED_MODEL

def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_collection("book_summaries")

def _embed_query(text: str, client: OpenAI):
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding

def semantic_search(query: str, k: int = 3, client: OpenAI | None = None) -> List[Dict]:
    if client is None:
        client = OpenAI(api_key=OPENAI_API_KEY)

    qemb = _embed_query(query, client)
    coll = get_collection()
    res = coll.query(
        query_embeddings=[qemb],   # <<< folosim embeddings, nu query_texts
        n_results=k,
        include=["metadatas", "documents", "distances"]
    )

    out = []
    for i in range(len(res["documents"][0])):
        meta = res["metadatas"][0][i]
        out.append({
            "title": meta.get("title", ""),
            "themes": meta.get("themes", ""),  # e string, nu listă
            "doc": res["documents"][0][i],
            "score": float(res["distances"][0][i]),
        })
    out.sort(key=lambda x: x["score"])
    return out
