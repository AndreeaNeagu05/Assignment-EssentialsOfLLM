import sys
from rich.console import Console
from rich.prompt import Prompt
from openai import OpenAI

from src.config import OPENAI_API_KEY, CHAT_MODEL, PROFANITY_LIST
from src.retriever import semantic_search
from src.tools import get_summary_by_title

console = Console()

SYSTEM_PROMPT = """You are Smart Librarian, a helpful book recommender.
Use the retrieved candidate books to pick ONE best recommendation.
Return exactly this JSON:
{"title": "...", "why": "..."} 
Where:
- title = exact book title from candidates
- why = 2-4 concise sentences referencing the user's interests.
If no good match, choose the closest.
"""


def contains_profanity(text: str) -> bool:
    low = text.lower()
    return any(bad in low for bad in PROFANITY_LIST)


def chat_once(user_query: str, client: OpenAI):
    # 1) Filtru limbaj nepotrivit (opțional)
    if contains_profanity(user_query):
        console.print("[yellow]Te rog formulează civilizat. Îți pot recomanda cărți cu drag![/yellow]")
        return

    # 2) RAG: căutare
    candidates = semantic_search(user_query, k=5, client=client)
    if not candidates:
        console.print("[red]Nu am găsit nimic în colecție.[/red]")
        return

    # 3) Construim mesaje către LLM, oferind candidații
    candidate_text = "\n".join([f"- {c['title']} | Teme: {c['themes']}" for c in candidates])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User query: {user_query}\nCandidates:\n{candidate_text}"}
    ]

    # 4) Obținem recomandarea (title + why)
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.3)
    raw = resp.choices[0].message.content.strip()

    # 5) Parse simplu JSON
    import json
    try:
        data = json.loads(raw)
        title = data.get("title") or candidates[0]["title"]
        why = data.get("why", "")
    except Exception:
        # fallback robust
        title = candidates[0]["title"]
        why = "Se potrivește cel mai bine cu temele interesului tău."

    # 6) Tool-calling local: get_summary_by_title
    summary = get_summary_by_title(title) or "(nu am găsit rezumatul complet local)"

    # 7) Afișare
    console.rule("[bold green]Recomandarea ta")
    console.print(f"[bold]Titlu:[/bold] {title}")
    console.print(f"[bold]De ce:[/bold] {why}")
    console.print("\n[bold]Rezumat detaliat[/bold]")
    console.print(summary)


def main():
    assert OPENAI_API_KEY, "Setează OPENAI_API_KEY în .env"
    client = OpenAI(api_key=OPENAI_API_KEY)
    console.print(
        "[bold cyan]Smart Librarian[/bold cyan] — scrie o întrebare (ex: 'Vreau o carte despre prietenie și magie').")
    console.print("Tastează 'exit' pentru a ieși.\n")
    while True:
        q = Prompt.ask("[bold magenta]Tu[/bold magenta]")
        if q.strip().lower() in {"exit", "quit"}:
            console.print("La revedere!")
            sys.exit(0)
        chat_once(q, client)


if __name__ == "__main__":
    main()
