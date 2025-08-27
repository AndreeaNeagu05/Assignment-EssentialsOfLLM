import json
from pathlib import Path
from typing import Optional

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "book_summaries.json"

def load_books():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_summary_by_title(title: str) -> Optional[str]:
    books = load_books()
    for b in books:
        if b["title"].strip().lower() == title.strip().lower():
            return b["full"]
    return None
