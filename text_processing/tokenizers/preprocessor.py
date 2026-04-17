import os

import jieba


DEFAULT_PUNCTUATION = {
    "\uff0c",
    "\u3002",
    "\uff1b",
    "\uff1a",
    "\uff01",
    "\uff1f",
    "\u201c",
    "\u201d",
    "\u3001",
    "\n",
    "\t",
    " ",
}


class TextPreprocessor:
    """Jieba-based tokenizer with stopword filtering and synonym normalization."""

    def __init__(self, stopwords_path=None, synonyms_path=None):
        self.stopwords = set()
        if stopwords_path and os.path.exists(stopwords_path):
            with open(stopwords_path, "r", encoding="utf-8") as file:
                self.stopwords = {line.strip() for line in file if line.strip()}

        self.synonyms = {}
        if synonyms_path and os.path.exists(synonyms_path):
            with open(synonyms_path, "r", encoding="utf-8") as file:
                for line in file:
                    words = [word.strip() for word in line.strip().split(",") if word.strip()]
                    if len(words) <= 1:
                        continue
                    base_word = words[0]
                    for word in words[1:]:
                        self.synonyms[word] = base_word

        self.punctuations = set(DEFAULT_PUNCTUATION)

    def clean_and_cut(self, text):
        result = []
        for word in jieba.cut(text or "", cut_all=False):
            word = word.strip()
            if not word or word in self.stopwords or word in self.punctuations:
                continue
            result.append(self.synonyms.get(word, word))
        return result
