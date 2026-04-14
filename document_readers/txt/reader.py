def read_txt_document(filepath: str, preview_mode: bool = False) -> str:
    """Read plain text with common Chinese/UTF-8 encodings."""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read().strip()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="gbk") as file:
            return file.read().strip()
