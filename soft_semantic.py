import os
from typing import Dict, Iterable, List, Optional

import numpy as np


class SoftSemanticScorer:
    """
    Lightweight semantic scorer for traditional mode.
    - Works with optional word vectors (.vec text format).
    - Falls back to synonym mapping when vectors are unavailable.
    """

    def __init__(
        self,
        embeddings_path: Optional[str] = None,
        synonyms_path: Optional[str] = None,
        similarity_threshold: float = 0.55,
        synonym_similarity: float = 0.92,
        max_terms: int = 220,
    ):
        self.embeddings_path = embeddings_path
        self.similarity_threshold = similarity_threshold
        self.synonym_similarity = synonym_similarity
        self.max_terms = max_terms

        self.word_vectors: Dict[str, np.ndarray] = {}
        self._sim_cache: Dict[str, float] = {}
        self._loaded_words = set()
        self._embedding_loaded = False

        self.last_vocab_size = 0
        self.last_vector_hits = 0
        self.last_vector_coverage = 0.0

        self.synonym_to_base = {}
        self._load_synonyms(synonyms_path)

    @property
    def semantic_enabled(self) -> bool:
        # semantic scoring is available if either vectors or synonym map exists
        has_vector_support = bool(self.embeddings_path and os.path.isfile(self.embeddings_path))
        has_synonyms = bool(self.synonym_to_base)
        return has_vector_support or has_synonyms

    def _load_synonyms(self, synonyms_path: Optional[str]) -> None:
        if not synonyms_path or not os.path.exists(synonyms_path):
            return

        try:
            with open(synonyms_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    words = [w.strip() for w in line.split(",") if w.strip()]
                    if not words:
                        continue
                    base = words[0]
                    self.synonym_to_base[base] = base
                    for w in words[1:]:
                        self.synonym_to_base[w] = base
        except Exception:
            # keep scorer robust even if synonym file contains bad encoding lines
            pass

    def _normalize_word(self, word: str) -> str:
        if word in self.synonym_to_base:
            return self.synonym_to_base[word]
        return word

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return any("\u4e00" <= ch <= "\u9fff" for ch in text)

    def _probe_vector_cjk_ratio(self, sample_lines: int = 5000) -> float:
        """
        Quick probe: inspect first N vector tokens to estimate whether file is CJK-oriented.
        """
        if not self.embeddings_path or not os.path.isfile(self.embeddings_path):
            return 0.0

        total = 0
        cjk = 0
        try:
            with open(self.embeddings_path, "r", encoding="utf-8", errors="ignore") as f:
                first = f.readline()
                if not self._is_header_line(first):
                    token = self._normalize_word_from_line(first)
                    if token:
                        total += 1
                        cjk += 1 if self._contains_cjk(token) else 0

                for line in f:
                    token = self._normalize_word_from_line(line)
                    if not token:
                        continue
                    total += 1
                    cjk += 1 if self._contains_cjk(token) else 0
                    if total >= sample_lines:
                        break
        except Exception:
            return 0.0

        return cjk / max(1, total)

    def _vocab_cjk_ratio(self, vocab_items: List[str]) -> float:
        if not vocab_items:
            return 0.0
        unique_vocab = set(vocab_items)
        cjk_count = sum(1 for w in unique_vocab if self._contains_cjk(w))
        return cjk_count / max(1, len(unique_vocab))

    def prepare_vocab(self, vocab_list: Iterable[str]) -> None:
        """
        Lazy-load only vectors used by current corpus vocabulary.
        This keeps memory footprint low when using large .vec files.
        """
        vocab_items = list(vocab_list)
        self.last_vocab_size = len(vocab_items)
        self.last_vector_hits = 0
        self.last_vector_coverage = 0.0

        if self._embedding_loaded:
            self._update_coverage_stats(vocab_items)
            return
        if not self.embeddings_path:
            if self.synonym_to_base:
                print(">>> [Semantic] Embedding path not set; running in synonym-only mode.")
            return

        if os.path.isdir(self.embeddings_path):
            print(f">>> [Semantic][Warn] Expected a .vec file but got directory: {self.embeddings_path}")
            return

        if not os.path.isfile(self.embeddings_path):
            if self.synonym_to_base:
                print(f">>> [Semantic] Embedding file not found: {self.embeddings_path}. Using synonym-only mode.")
            return

        vocab_cjk_ratio = self._vocab_cjk_ratio(vocab_items)
        if vocab_cjk_ratio > 0.3:
            sampled_cjk_ratio = self._probe_vector_cjk_ratio()
            if sampled_cjk_ratio < 0.005:
                print(
                    ">>> [Semantic][Warn] Embedding file seems language-mismatched "
                    f"(vocab CJK={vocab_cjk_ratio * 100:.2f}%, vector sample CJK={sampled_cjk_ratio * 100:.2f}%). "
                    "Skipping vector scan for this session."
                )
                self._embedding_loaded = True
                return

        target_words = set(vocab_items)
        # include synonym base words for better coverage
        target_words.update(self.synonym_to_base.keys())
        target_words.update(self.synonym_to_base.values())

        self._load_vectors_subset(target_words)
        self._embedding_loaded = True
        self._update_coverage_stats(vocab_items)

    def _update_coverage_stats(self, vocab_items: List[str]) -> None:
        if not vocab_items:
            self.last_vector_hits = 0
            self.last_vector_coverage = 0.0
            return

        hits = 0
        unique_vocab = set(vocab_items)
        for word in unique_vocab:
            mapped = self._normalize_word(word)
            if mapped in self.word_vectors:
                hits += 1

        self.last_vector_hits = hits
        self.last_vector_coverage = hits / max(1, len(unique_vocab))

        if self.word_vectors:
            print(
                ">>> [Semantic] Vector coverage: "
                f"{self.last_vector_hits}/{len(unique_vocab)} "
                f"({self.last_vector_coverage * 100:.2f}%)."
            )
            # A very low hit ratio often means wrong language vectors or wrong file.
            if len(unique_vocab) >= 80 and self.last_vector_coverage < 0.03:
                print(
                    ">>> [Semantic][Warn] Coverage is extremely low. "
                    "Check whether the embedding file language matches your corpus."
                )

    def _load_vectors_subset(self, target_words: set) -> None:
        loaded = 0
        try:
            with open(self.embeddings_path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
                # fastText .vec may start with "<count> <dim>"
                # if it is not a header, process it as a normal vector line
                if not self._is_header_line(first_line):
                    self._try_parse_vector_line(first_line, target_words)
                    loaded += 1 if self._normalize_word_from_line(first_line) in self.word_vectors else 0

                for line in f:
                    word = self._normalize_word_from_line(line)
                    if word is None or word not in target_words or word in self.word_vectors:
                        continue
                    if self._try_parse_vector_line(line, target_words):
                        loaded += 1
        except Exception:
            # fallback silently; scorer can still use synonym mapping
            pass

        if loaded:
            print(f">>> [Semantic] Loaded {loaded} embedding vectors for current vocabulary.")

    @staticmethod
    def _is_header_line(line: str) -> bool:
        parts = line.strip().split()
        if len(parts) != 2:
            return False
        return parts[0].isdigit() and parts[1].isdigit()

    @staticmethod
    def _normalize_word_from_line(line: str) -> Optional[str]:
        parts = line.strip().split()
        if len(parts) < 3:
            return None
        return parts[0]

    def _try_parse_vector_line(self, line: str, target_words: set) -> bool:
        parts = line.strip().split()
        if len(parts) < 3:
            return False
        word = parts[0]
        if word not in target_words:
            return False
        try:
            vec = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
        except ValueError:
            return False

        norm = np.linalg.norm(vec)
        if norm <= 0:
            return False
        self.word_vectors[word] = vec / norm
        return True

    def _pair_cache_key(self, a: str, b: str) -> str:
        return f"{a}\t{b}" if a <= b else f"{b}\t{a}"

    def _term_similarity(self, word_a: str, word_b: str) -> float:
        if word_a == word_b:
            return 1.0

        norm_a = self._normalize_word(word_a)
        norm_b = self._normalize_word(word_b)
        if norm_a == norm_b:
            return self.synonym_similarity

        key = self._pair_cache_key(norm_a, norm_b)
        if key in self._sim_cache:
            return self._sim_cache[key]

        vec_a = self.word_vectors.get(norm_a)
        vec_b = self.word_vectors.get(norm_b)
        if vec_a is None or vec_b is None:
            self._sim_cache[key] = 0.0
            return 0.0

        sim = float(np.dot(vec_a, vec_b))
        if sim < self.similarity_threshold:
            sim = 0.0
        self._sim_cache[key] = sim
        return sim

    def _select_top_terms(self, vector: np.ndarray) -> np.ndarray:
        indices = np.flatnonzero(vector)
        if len(indices) <= self.max_terms:
            return indices

        values = np.abs(vector[indices])
        top_pos = np.argpartition(values, -self.max_terms)[-self.max_terms :]
        return indices[top_pos]

    def _soft_norm(self, indices: np.ndarray, values: np.ndarray, vocab_list: List[str]) -> float:
        if len(indices) == 0:
            return 0.0

        norm_value = 0.0
        for i in indices:
            norm_value += float(values[i] * values[i])

        # off-diagonal similarity terms
        idx_list = indices.tolist()
        for p in range(len(idx_list)):
            i = idx_list[p]
            wi = vocab_list[i]
            xi = float(values[i])
            for q in range(p + 1, len(idx_list)):
                j = idx_list[q]
                sim = self._term_similarity(wi, vocab_list[j])
                if sim <= 0:
                    continue
                norm_value += 2.0 * xi * float(values[j]) * sim
        return norm_value

    def score(self, vec_a: np.ndarray, vec_b: np.ndarray, vocab_list: List[str]) -> float:
        if len(vec_a) != len(vec_b) or len(vec_a) != len(vocab_list):
            return 0.0

        idx_a = self._select_top_terms(vec_a)
        idx_b = self._select_top_terms(vec_b)
        if len(idx_a) == 0 or len(idx_b) == 0:
            return 0.0

        numerator = 0.0
        for i in idx_a:
            wi = vocab_list[i]
            xi = float(vec_a[i])
            for j in idx_b:
                yj = float(vec_b[j])
                if i == j:
                    continue
                sim = self._term_similarity(wi, vocab_list[j])
                if sim <= 0:
                    continue
                numerator += xi * yj * sim

        norm_a = self._soft_norm(idx_a, vec_a, vocab_list)
        norm_b = self._soft_norm(idx_b, vec_b, vocab_list)
        if norm_a <= 0 or norm_b <= 0:
            return 0.0

        score = numerator / float(np.sqrt(norm_a * norm_b))
        # numeric guard
        if score < 0:
            return 0.0
        if score > 1:
            return 1.0
        return float(score)
