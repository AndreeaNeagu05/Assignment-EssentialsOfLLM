# Smart Librarian – AI cu RAG + Tool Completion

## Cerințe acoperite
- Vector store: **ChromaDB** (nu OpenAI vector store)
- Embeddings: **text-embedding-3-small**
- RAG: căutare semantică după teme/context
- Tool local: `get_summary_by_title(title: str)` -> returnează rezumat complet
- Interfață: **CLI** (Streamlit opțional)
- Filtru limbaj nepotrivit (simplu)
- (Opțional) TTS/STT/Image gen – vezi mai jos

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# editează .env și pune OPENAI_API_KEY
