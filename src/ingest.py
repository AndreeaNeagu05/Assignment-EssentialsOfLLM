import json
from pathlib import Path
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from src.config import OPENAI_API_KEY, EMBED_MODEL, CHROMA_DIR

DATA = Path(__file__).resolve().parents[1] / "data" / "book_summaries.json"

def embed_texts(texts, client):
    # Embeddings batch-friendly
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def main():
    assert OPENAI_API_KEY, "Setează OPENAI_API_KEY în .env"
    client = OpenAI(api_key=OPENAI_API_KEY)

    with open(DATA, "r", encoding="utf-8") as f:
        books = json.load(f)

    # Construim documentele: folosim short + themes pentru căutare semantică
    ids, docs, metas = [], [], []
    for i, b in enumerate(books):
        ids.append(str(i))
        docs.append(f"{b['title']}\n{b['short']}\nTeme: {', '.join(b['themes'])}")
        #metas.append({"title": b["title"], "themes": b["themes"]})
        metas.append({"title": b["title"], "themes": ", ".join(b["themes"])})

    embeddings = embed_texts(docs, client)

    chroma = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )
    chroma.reset()  # curățăm orice colecție anterioară

    coll = chroma.get_or_create_collection(name="book_summaries")
    coll.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    print("Ingest complet în ChromaDB.")

if __name__ == "__main__":
    main()
