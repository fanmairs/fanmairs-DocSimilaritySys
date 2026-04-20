from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DICTS_DIR = PROJECT_ROOT / "dicts"
TEMP_UPLOAD_DIR = PROJECT_ROOT / "temp_uploads"
TASK_DB_FILE = PROJECT_ROOT / "tasks.db"

STOPWORDS_PATH = DICTS_DIR / "stopwords.txt"
SYNONYMS_PATH = DICTS_DIR / "synonyms.txt"
SEMANTIC_EMBEDDINGS_PATH = DICTS_DIR / "embeddings" / "fasttext_zh.vec"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
