import math
import re

import numpy as np
from typing import Dict, List, Optional, Tuple


class DeepSemanticEngine:
    def __init__(self, model_name='BAAI/bge-large-zh-v1.5'):
        """Deep semantic matching engine powered by BGE."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers not found. Please run: pip install sentence-transformers torch")

        import torch

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        self.tokenizer = self._resolve_tokenizer()

        if self.device == 'cuda':
            print(">>> [System] CUDA detected. Converting BGE model to FP16 for lower VRAM and higher throughput.")
            self.model = self.model.half()

        model_max_tokens = getattr(self.model, "max_seq_length", None)
        if not isinstance(model_max_tokens, int) or model_max_tokens <= 0:
            model_max_tokens = 512
        self.model_max_tokens = model_max_tokens
        self.encode_overlap_tokens = min(96, max(32, self.model_max_tokens // 6))
        self.window_max_tokens = 96
        self.window_min_tokens = 24
        self.window_overlap_tokens = 24
        self.window_topk = 3
        self.window_min_chars = 10
        self._token_count_cache = {}

        # Target window cache: reused across many reference documents in one task.
        self._cache_target_text = None
        self._cache_win1 = None
        self._cache_emb1 = None

        # Target document semantic cache for Step6 scoring.
        self._cache_target_doc_text = None
        self._cache_target_doc_embedding = None
        self._cache_target_doc_paragraphs = None
        self._cache_target_para_embeddings = None

        # Threshold profiles for different business scenarios.
        # strict   -> final adjudication, minimize false positives
        # balanced -> daily default
        # recall   -> clue mining, minimize misses
        self.default_profile = "balanced"
        self.threshold_profiles = {
            "strict": {
                "short_text_max_chars": 200,
                "outlier_std_k": 2.3,
                "outlier_percentile": 0.98,
                "outlier_percentile_margin": 0.012,
                "outlier_percentile_min_windows": 12,
                "short_low": 0.66,
                "short_high": 0.74,
                "long_low": 0.86,
                "long_high": 0.91,
                "paragraph_threshold": 0.97,
                "min_window_chars": 12,
                "score_weights": {
                    "doc_semantic": 0.20,
                    "coverage": 0.30,
                    "confidence": 0.50,
                },
                "final_score": {
                    "semantic_weight": 0.24,
                    "evidence_weight": 0.76,
                    "semantic_center": 0.84,
                    "semantic_scale": 0.07,
                    "coverage_gain": 9.0,
                    "low_evidence_cap_base": 0.10,
                    "low_evidence_cap_gain": 0.16,
                    "continuity_boost": 0.05,
                },
                "semantic_floor": 0.80,
                "score_gate": {
                    "base": 0.08,
                    "coverage": 0.55,
                    "confidence": 0.30,
                    "evidence": 0.25,
                    "low_cov_th": 0.05,
                    "mid_cov_th": 0.08,
                    "mid_conf_th": 0.35,
                    "mid_cov_cap": 0.28,
                    "topic_cov_th": 0.12,
                    "topic_conf_th": 0.75,
                    "topic_cap": 0.20,
                    "low_evidence_cov_th": 0.10,
                    "low_evidence_conf_th": 0.60,
                    "low_evidence_cap": 0.08,
                },
            },
            "balanced": {
                "short_text_max_chars": 200,
                "outlier_std_k": 2.0,
                "outlier_percentile": 0.95,
                "outlier_percentile_margin": 0.010,
                "outlier_percentile_min_windows": 10,
                "short_low": 0.60,
                "short_high": 0.70,
                "long_low": 0.82,
                "long_high": 0.88,
                "paragraph_threshold": 0.95,
                "min_window_chars": 10,
                "score_weights": {
                    "doc_semantic": 0.18,
                    "coverage": 0.52,
                    "confidence": 0.30,
                },
                "final_score": {
                    "semantic_weight": 0.30,
                    "evidence_weight": 0.70,
                    "semantic_center": 0.80,
                    "semantic_scale": 0.08,
                    "coverage_gain": 8.0,
                    "low_evidence_cap_base": 0.14,
                    "low_evidence_cap_gain": 0.20,
                    "continuity_boost": 0.06,
                },
                "semantic_floor": 0.86,
                "score_gate": {
                    "base": 0.09,
                    "coverage": 0.60,
                    "confidence": 0.22,
                    "evidence": 0.22,
                    "low_cov_th": 0.05,
                    "low_cov_cap": 0.20,
                    "mid_cov_th": 0.08,
                    "mid_conf_th": 0.35,
                    "mid_cov_cap": 0.30,
                    "topic_cov_th": 0.15,
                    "topic_conf_th": 0.75,
                    "topic_cap": 0.24,
                    "low_evidence_cov_th": 0.10,
                    "low_evidence_conf_th": 0.55,
                    "low_evidence_cap": 0.14,
                },
            },
            "recall": {
                "short_text_max_chars": 200,
                "outlier_std_k": 1.7,
                "outlier_percentile": 0.93,
                "outlier_percentile_margin": 0.008,
                "outlier_percentile_min_windows": 8,
                "short_low": 0.55,
                "short_high": 0.65,
                "long_low": 0.78,
                "long_high": 0.84,
                "paragraph_threshold": 0.92,
                "min_window_chars": 8,
                "score_weights": {
                    "doc_semantic": 0.35,
                    "coverage": 0.40,
                    "confidence": 0.25,
                },
                "final_score": {
                    "semantic_weight": 0.36,
                    "evidence_weight": 0.64,
                    "semantic_center": 0.76,
                    "semantic_scale": 0.09,
                    "coverage_gain": 7.0,
                    "low_evidence_cap_base": 0.18,
                    "low_evidence_cap_gain": 0.22,
                    "continuity_boost": 0.07,
                },
                "semantic_floor": 0.74,
                "score_gate": {
                    "base": 0.15,
                    "coverage": 0.50,
                    "confidence": 0.25,
                    "evidence": 0.20,
                    "low_cov_th": 0.04,
                    "low_cov_cap": 0.22,
                    "mid_cov_th": 0.07,
                    "mid_conf_th": 0.30,
                    "mid_cov_cap": 0.30,
                    "topic_cov_th": 0.10,
                    "topic_conf_th": 0.70,
                    "topic_cap": 0.26,
                },
            },
        }

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            exp_term = math.exp(-value)
            return float(1.0 / (1.0 + exp_term))

        exp_term = math.exp(value)
        return float(exp_term / (1.0 + exp_term))

    @staticmethod
    def _normalize_text(text: str) -> str:
        import re

        return re.sub(r'\s+', ' ', text or '').strip()

    @staticmethod
    def _normalize_for_paragraphs(text: str) -> str:
        import re

        normalized = (text or '').replace('\r\n', '\n').replace('\r', '\n')
        normalized = re.sub(r'[ \t\f\v]+', ' ', normalized)
        normalized = re.sub(r'\n{3,}', '\n\n', normalized)
        return normalized.strip()

    def _resolve_profile(self, profile_name: Optional[str]) -> Tuple[str, Dict]:
        if not isinstance(profile_name, str) or not profile_name.strip():
            return self.default_profile, self.threshold_profiles[self.default_profile]

        normalized = profile_name.strip().lower()
        if normalized not in self.threshold_profiles:
            print(f">>> [BGE][Warn] Unknown threshold profile: {profile_name}. Fallback to {self.default_profile}.")
            normalized = self.default_profile
        return normalized, self.threshold_profiles[normalized]

    def _resolve_tokenizer(self):
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer

        try:
            return getattr(self.model._first_module(), "tokenizer", None)
        except Exception:
            return None

    @staticmethod
    def _make_span(text: str, start: int, end: Optional[int] = None) -> Optional[Dict]:
        raw = text or ""
        if end is None:
            end = start + len(raw)

        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw) - len(raw.rstrip())
        span_start = start + left_trim
        span_end = end - right_trim
        if span_end <= span_start:
            return None

        cleaned = raw.strip()
        if not cleaned:
            return None

        return {"text": cleaned, "start": span_start, "end": span_end}

    def _estimate_token_count(self, text: str) -> int:
        normalized = (text or "").strip()
        if not normalized:
            return 0

        cached = self._token_count_cache.get(normalized)
        if cached is not None:
            return cached

        count = None
        if self.tokenizer is not None:
            try:
                count = len(self.tokenizer.encode(normalized, add_special_tokens=False, truncation=False))
            except TypeError:
                try:
                    encoded = self.tokenizer(
                        normalized,
                        add_special_tokens=False,
                        truncation=False,
                        return_attention_mask=False,
                    )
                    input_ids = encoded.get("input_ids", [])
                    if input_ids and isinstance(input_ids[0], list):
                        input_ids = input_ids[0]
                    count = len(input_ids)
                except Exception:
                    count = None
            except Exception:
                count = None

        if count is None:
            cjk_count = len(re.findall(r'[\u4e00-\u9fff]', normalized))
            latin_count = len(re.findall(r'[A-Za-z0-9_]+', normalized))
            punctuation_count = len(re.findall(r'[^\w\s]', normalized))
            count = cjk_count + latin_count + max(1, punctuation_count // 2)

        count = max(1, int(count))
        self._token_count_cache[normalized] = count
        return count

    def _tokenize_with_offsets(self, text: str) -> Optional[List[Tuple[int, int]]]:
        normalized = text or ""
        if not normalized or self.tokenizer is None:
            return None

        try:
            encoded = self.tokenizer(
                normalized,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                return_offsets_mapping=True,
            )
        except TypeError:
            try:
                encoded = self.tokenizer(
                    normalized,
                    add_special_tokens=False,
                    truncation=False,
                    return_offsets_mapping=True,
                )
            except Exception:
                return None
        except Exception:
            return None

        offsets = encoded.get("offset_mapping")
        if not offsets:
            return None

        if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
            offsets = offsets[0]

        cleaned = []
        for pair in offsets:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            start, end = int(pair[0]), int(pair[1])
            if end > start:
                cleaned.append((start, end))
        return cleaned or None

    def _split_clauses_with_offsets(self, text: str, base_start: int) -> List[Dict[str, int]]:
        if not text:
            return []

        pattern = re.compile(r'[^\uff0c,\u3001\uff1a:]+[\uff0c,\u3001\uff1a:]*')
        clauses = []
        for match in pattern.finditer(text):
            span = self._make_span(match.group(0), base_start + match.start(), base_start + match.end())
            if span is not None:
                clauses.append(span)

        if clauses:
            return clauses

        fallback = self._make_span(text, base_start, base_start + len(text))
        return [fallback] if fallback else []

    def _slice_text_by_token_budget(
        self,
        text: str,
        base_start: int,
        max_tokens: int,
        overlap_tokens: int,
    ) -> List[Dict]:
        span = self._make_span(text, base_start, base_start + len(text or ""))
        if span is None:
            return []

        text = span["text"]
        base_start = span["start"]
        offsets = self._tokenize_with_offsets(text)

        if offsets:
            pieces = []
            cursor = 0
            total_tokens = len(offsets)
            while cursor < total_tokens:
                end = min(total_tokens, cursor + max_tokens)
                start_char = offsets[cursor][0]
                end_char = offsets[end - 1][1]
                piece = self._make_span(text[start_char:end_char], base_start + start_char, base_start + end_char)
                if piece is not None:
                    piece["tokens"] = end - cursor
                    pieces.append(piece)

                if end >= total_tokens:
                    break

                next_cursor = end - overlap_tokens
                if next_cursor <= cursor:
                    next_cursor = cursor + 1
                cursor = next_cursor
            return pieces

        approx_chars = max(24, int(max_tokens * 1.6))
        overlap_chars = max(0, int(overlap_tokens * 1.6))
        pieces = []
        cursor = 0
        total_chars = len(text)

        while cursor < total_chars:
            end = min(total_chars, cursor + approx_chars)
            if end < total_chars:
                tail = text[cursor:end]
                cut_points = [tail.rfind(ch) for ch in "，,、：: "]
                cut_points = [tail.rfind(ch) for ch in (",", ":", " ", "\u3001", "\uff0c", "\uff1a")]
                best_cut = max(cut_points)
                if best_cut > approx_chars * 0.6:
                    end = cursor + best_cut + 1

            piece = self._make_span(text[cursor:end], base_start + cursor, base_start + end)
            if piece is not None:
                piece["tokens"] = self._estimate_token_count(piece["text"])
                pieces.append(piece)

            if end >= total_chars:
                break

            cursor = max(cursor + 1, end - overlap_chars)

        return pieces

    def _fit_span_to_token_budget(
        self,
        text: str,
        base_start: int,
        max_tokens: int,
        overlap_tokens: Optional[int] = None,
    ) -> List[Dict]:
        span = self._make_span(text, base_start, base_start + len(text or ""))
        if span is None:
            return []

        token_count = self._estimate_token_count(span["text"])
        if token_count <= max_tokens:
            span["tokens"] = token_count
            return [span]

        slice_overlap = overlap_tokens or max(8, max_tokens // 4)
        clauses = self._split_clauses_with_offsets(span["text"], span["start"])
        if len(clauses) > 1:
            pieces = []
            for clause in clauses:
                clause_tokens = self._estimate_token_count(clause["text"])
                if clause_tokens <= max_tokens:
                    clause["tokens"] = clause_tokens
                    pieces.append(clause)
                else:
                    pieces.extend(
                        self._slice_text_by_token_budget(
                            clause["text"],
                            clause["start"],
                            max_tokens=max_tokens,
                            overlap_tokens=slice_overlap,
                        )
                    )
            if pieces:
                return pieces

        return self._slice_text_by_token_budget(
            span["text"],
            span["start"],
            max_tokens=max_tokens,
            overlap_tokens=slice_overlap,
        )

    def _prepare_text_units(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: Optional[int] = None,
    ) -> List[Dict]:
        sentences = self._split_sentences_with_offsets(text)
        if not sentences:
            return self._fit_span_to_token_budget(text, 0, max_tokens, overlap_tokens)

        units = []
        for sentence in sentences:
            units.extend(
                self._fit_span_to_token_budget(
                    sentence["text"],
                    sentence["start"],
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
        return units

    def _build_text_windows(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int,
        min_tokens: int = 1,
        min_chars: int = 1,
    ) -> List[Dict]:
        units = self._prepare_text_units(
            text,
            max_tokens=max(8, int(max_tokens)),
            overlap_tokens=max(0, int(overlap_tokens)),
        )
        if not units:
            return []

        windows = []
        seen = set()
        cursor = 0
        total_units = len(units)

        while cursor < total_units:
            end = cursor
            total_tokens = 0
            while end < total_units:
                unit_tokens = max(1, int(units[end].get("tokens", 0)) or self._estimate_token_count(units[end]["text"]))
                if end > cursor and total_tokens >= min_tokens and total_tokens + unit_tokens > max_tokens:
                    break
                total_tokens += unit_tokens
                end += 1
                if total_tokens >= max_tokens:
                    break

            if end == cursor:
                end = cursor + 1
                total_tokens = max(1, int(units[cursor].get("tokens", 1)))

            window = self._make_span(
                text[units[cursor]["start"]:units[end - 1]["end"]],
                units[cursor]["start"],
                units[end - 1]["end"],
            )
            if window is not None:
                window["tokens"] = total_tokens
                key = (window["start"], window["end"])
                if len(window["text"]) >= min_chars and key not in seen:
                    windows.append(window)
                    seen.add(key)

            if end >= total_units:
                break

            next_cursor = end
            overlapped_tokens = 0
            while next_cursor > cursor + 1 and overlapped_tokens < overlap_tokens:
                next_cursor -= 1
                overlapped_tokens += max(1, int(units[next_cursor].get("tokens", 1)))

            if next_cursor <= cursor:
                next_cursor = cursor + 1
            cursor = next_cursor

        if windows:
            return windows

        fallback = self._make_span(text, 0, len(text or ""))
        if fallback is None:
            return []
        fallback["tokens"] = self._estimate_token_count(fallback["text"])
        return [fallback]

    def encode(self, texts, max_length=500):
        """
        Encode text list into embeddings.
        Long text is chunked on sentence boundaries with overlapping token windows
        to avoid BERT max-token truncation bias.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([])

        token_budget = max(32, min(int(max_length), int(self.model_max_tokens)))
        results = [None] * len(texts)

        short_texts = []
        short_indices = []
        long_texts = []
        long_indices = []

        for i, text in enumerate(texts):
            if self._estimate_token_count(text) <= token_budget:
                short_texts.append(text)
                short_indices.append(i)
            else:
                long_texts.append(text)
                long_indices.append(i)

        if short_texts:
            show_bar = len(short_texts) > 50
            short_embs = self.model.encode(
                short_texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=show_bar,
            )
            for idx, emb in zip(short_indices, short_embs):
                results[idx] = emb

        if long_texts:
            chunk_overlap = min(self.encode_overlap_tokens, max(16, token_budget // 6))
            for idx, text in zip(long_indices, long_texts):
                chunks = [
                    chunk["text"]
                    for chunk in self._build_text_windows(
                        text,
                        max_tokens=token_budget,
                        overlap_tokens=chunk_overlap,
                        min_tokens=max(32, token_budget // 2),
                        min_chars=20,
                    )
                ]
                if not chunks:
                    chunks = [text]
                chunk_embeddings = self.model.encode(
                    chunks,
                    batch_size=16,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                pooled_embedding = np.mean(chunk_embeddings, axis=0)
                norm = np.linalg.norm(pooled_embedding)
                if norm > 0:
                    pooled_embedding = pooled_embedding / norm
                results[idx] = pooled_embedding

        return np.array(results)

    def calculate_similarity(self, vec1, vec2):
        """Cosine similarity for normalized vectors (dot product)."""
        return float(np.dot(vec1, vec2))

    # Legacy helpers retained temporarily for reference during the Step1 rollout.
    def _legacy_split_sentences_with_offsets_unused(self, text: str) -> List[Dict[str, int]]:
        import re

        if not text:
            return []

        pattern = re.compile(r'[^。！？；\n]+[。！？；\n]*')
        sentences = []
        for match in pattern.finditer(text):
            raw = match.group(0)
            if not raw:
                continue
            left_trim = len(raw) - len(raw.lstrip())
            right_trim = len(raw) - len(raw.rstrip())
            start = match.start() + left_trim
            end = match.end() - right_trim
            if end <= start:
                continue
            seg = text[start:end]
            if seg.strip():
                sentences.append({"text": seg.strip(), "start": start, "end": end})

        if not sentences and text.strip():
            stripped = text.strip()
            start = text.find(stripped)
            return [{"text": stripped, "start": start, "end": start + len(stripped)}]

        return sentences

    def _legacy_build_windows_unused(self, text: str) -> List[Dict]:
        import re

        sentences = self._split_sentences_with_offsets(text)
        if len(sentences) <= 1:
            if text.strip():
                stripped = text.strip()
                start = text.find(stripped)
                return [{"text": stripped, "start": start, "end": start + len(stripped)}]
            return []

        windows = []
        for i in range(len(sentences) - 1):
            start = sentences[i]["start"]
            end = sentences[i + 1]["end"]
            if end <= start:
                continue
            win_text = text[start:end].strip()
            if not win_text:
                continue

            chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', win_text))
            if len(win_text) > 5 and (chinese_chars / max(len(win_text), 1) > 0.2):
                windows.append({"text": win_text, "start": start, "end": end})

        if windows:
            return windows

        stripped = text.strip()
        if not stripped:
            return []
        start = text.find(stripped)
        return [{"text": stripped, "start": start, "end": start + len(stripped)}]

    def _split_sentences_with_offsets(self, text: str) -> List[Dict[str, int]]:
        if not text:
            return []

        pattern = re.compile(r'[^\u3002\uff01\uff1f\uff1b!?;\n]+[\u3002\uff01\uff1f\uff1b!?;\n]*')
        sentences = []
        for match in pattern.finditer(text):
            span = self._make_span(match.group(0), match.start(), match.end())
            if span is not None:
                sentences.append(span)

        if sentences:
            return sentences

        fallback = self._make_span(text, 0, len(text))
        return [fallback] if fallback else []

    def _build_windows(self, text: str) -> List[Dict]:
        windows = self._build_text_windows(
            text,
            max_tokens=self.window_max_tokens,
            overlap_tokens=self.window_overlap_tokens,
            min_tokens=self.window_min_tokens,
            min_chars=self.window_min_chars,
        )
        if not windows:
            return []

        filtered = []
        for window in windows:
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', window["text"]))
            if len(window["text"]) <= 5:
                continue
            if chinese_chars > 0 and chinese_chars / max(len(window["text"]), 1) <= 0.2:
                continue
            filtered.append(window)

        return filtered or windows

    @staticmethod
    def _select_topk_indices(scores: np.ndarray, topk: int) -> np.ndarray:
        if scores.size == 0 or topk <= 0:
            return np.asarray([], dtype=int)

        k = min(int(topk), int(scores.size))
        if k == scores.size:
            return np.argsort(scores)[::-1]

        candidate_idx = np.argpartition(scores, -k)[-k:]
        return candidate_idx[np.argsort(scores[candidate_idx])[::-1]]

    def _resolve_outlier_metrics(self, sims: np.ndarray, peak_sim: float, profile_cfg: Dict) -> Dict[str, float]:
        if sims.size == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "std_threshold": 0.0,
                "percentile_threshold": 0.0,
                "effective_threshold": 0.0,
                "is_statistical_outlier": False,
                "is_percentile_outlier": False,
                "used_percentile": False,
            }

        mean_sim = float(np.mean(sims))
        std_sim = float(np.std(sims))
        outlier_std_k = float(profile_cfg.get("outlier_std_k", 2.0))
        std_threshold = mean_sim + outlier_std_k * std_sim

        percentile_cfg = float(profile_cfg.get("outlier_percentile", 0.0))
        if percentile_cfg > 1.0:
            percentile_cfg /= 100.0
        percentile_cfg = min(max(percentile_cfg, 0.0), 0.999)
        percentile_margin = float(profile_cfg.get("outlier_percentile_margin", 0.0))
        percentile_min_windows = max(1, int(profile_cfg.get("outlier_percentile_min_windows", 1)))

        percentile_threshold = 0.0
        used_percentile = percentile_cfg > 0.0 and sims.size >= percentile_min_windows
        is_percentile_outlier = False
        if used_percentile:
            percentile_threshold = float(np.quantile(sims, percentile_cfg)) + percentile_margin
            is_percentile_outlier = peak_sim >= percentile_threshold

        is_statistical_outlier = peak_sim > std_threshold
        effective_threshold = std_threshold
        if used_percentile:
            effective_threshold = min(std_threshold, percentile_threshold)

        return {
            "mean": mean_sim,
            "std": std_sim,
            "std_threshold": float(std_threshold),
            "percentile_threshold": float(percentile_threshold),
            "effective_threshold": float(effective_threshold),
            "is_statistical_outlier": bool(is_statistical_outlier),
            "is_percentile_outlier": bool(is_percentile_outlier),
            "used_percentile": bool(used_percentile),
        }

    def _score_window_candidate(
        self,
        item1: Dict,
        item2: Dict,
        raw_sim: float,
        outlier_threshold: float,
        profile_cfg: Dict,
    ) -> Optional[Dict]:
        import difflib

        text1 = item1["text"]
        text2 = item2["text"]

        min_window_chars = profile_cfg["min_window_chars"]
        if len(text1) <= min_window_chars or len(text2) <= min_window_chars:
            return None

        digit_ratio1 = sum(c.isdigit() for c in text1) / max(1, len(text1))
        digit_ratio2 = sum(c.isdigit() for c in text2) / max(1, len(text2))
        eng_ratio1 = len(re.findall(r'[a-zA-Z]', text1)) / max(1, len(text1))

        edit_similarity = difflib.SequenceMatcher(None, text1, text2).ratio()

        entities1 = self._extract_entities(text1)
        entities2 = self._extract_entities(text2)
        entity_iou = self._safe_iou(entities1, entities2)

        tags1 = self._extract_tags(text1)
        tags2 = self._extract_tags(text2)
        tag_iou = self._safe_iou(tags1, tags2)

        skeleton1 = self._get_skeleton(text1)
        skeleton2 = self._get_skeleton(text2)
        skeleton_sim = difflib.SequenceMatcher(None, skeleton1, skeleton2).ratio()

        rule_penalty = 1.0
        rule_flags = []

        if entity_iou < 0.1 and tag_iou < 0.25 and skeleton_sim > 0.8:
            rule_penalty *= 0.55
            rule_flags.append("template_skeleton")

        if self._is_formula_explanation(text1) and self._is_formula_explanation(text2):
            if entity_iou > 0.3 and edit_similarity < 0.8:
                rule_penalty *= 0.70
                rule_flags.append("formula_explanation")

        if entity_iou < 0.08:
            rule_penalty *= 0.88
            rule_flags.append("entity_mismatch")
        elif entity_iou < 0.16:
            rule_penalty *= 0.93

        if tag_iou < 0.10:
            rule_penalty *= 0.92
            rule_flags.append("tag_mismatch")

        # Keep technical / formula-heavy spans as downgraded evidence instead of hard rejecting them.
        # This matches the Step5 goal: prefer penalty over silent discard on boundary samples.
        if text1.count('.') >= 5:
            rule_penalty *= 0.92
            rule_flags.append("target_many_periods")
        if text2.count('.') >= 5:
            rule_penalty *= 0.92
            rule_flags.append("ref_many_periods")
        if digit_ratio1 >= 0.2:
            rule_penalty *= 0.84
            rule_flags.append("target_digit_heavy")
        if digit_ratio2 >= 0.2:
            rule_penalty *= 0.84
            rule_flags.append("ref_digit_heavy")
        if eng_ratio1 >= 0.4:
            rule_penalty *= 0.88
            rule_flags.append("target_english_heavy")

        eng_ratio2 = len(re.findall(r'[a-zA-Z]', text2)) / max(1, len(text2))
        if eng_ratio2 >= 0.4:
            rule_penalty *= 0.88
            rule_flags.append("ref_english_heavy")

        margin = raw_sim - outlier_threshold
        margin_norm = self._clamp01((margin + 0.05) / 0.20)

        confidence = raw_sim * (0.65 + 0.35 * margin_norm)
        confidence *= (0.90 + 0.10 * entity_iou)
        confidence *= (0.90 + 0.10 * tag_iou)
        confidence *= rule_penalty
        if edit_similarity < 0.12:
            confidence *= 0.90

        effective_score = self._clamp01(raw_sim * (0.55 + 0.45 * rule_penalty))
        return {
            'target_part': text1,
            'ref_part': text2,
            'score': float(effective_score),
            'raw_score': float(raw_sim),
            'confidence': float(self._clamp01(confidence)),
            'length': len(text1),
            'target_start': int(item1['start']),
            'target_end': int(item1['end']),
            'ref_start': int(item2['start']),
            'ref_end': int(item2['end']),
            'match_type': 'window',
            'rule_penalty': float(self._clamp01(rule_penalty)),
            'rule_flags': rule_flags,
        }

    @staticmethod
    def _extract_entities(text: str):
        import re

        return set(re.findall(r'[A-Za-z]+|\d+(?:\.\d+)?', text))

    @staticmethod
    def _extract_tags(text: str):
        import jieba.analyse

        return set(jieba.analyse.extract_tags(text, topK=5))

    @staticmethod
    def _get_skeleton(text: str) -> str:
        import jieba.posseg as pseg

        words = pseg.cut(text)
        skeleton = [w.word for w in words if w.flag in ['v', 'p', 'c', 'd']]
        return "".join(skeleton)

    @staticmethod
    def _is_formula_explanation(text: str) -> bool:
        import re

        explanation_keywords = ['公式', '其中', '表示', '定义', '计算', '如图', '如式', '等于', '获得', '所示']
        keyword_count = sum(1 for kw in explanation_keywords if kw in text)
        has_math_symbols = len(re.findall(r'[A-Za-z]+|\d+', text)) > 5
        return keyword_count >= 2 and has_math_symbols

    @staticmethod
    def _safe_iou(set_a: set, set_b: set, default: float = 1.0) -> float:
        if not set_a and not set_b:
            return default
        union = len(set_a.union(set_b))
        if union == 0:
            return default
        return len(set_a.intersection(set_b)) / union

    @staticmethod
    def _get_paragraphs(text: str, min_chars: int = 50, max_count: int = 24) -> List[str]:
        normalized = DeepSemanticEngine._normalize_for_paragraphs(text)
        if not normalized:
            return []

        paras = [p.strip() for p in normalized.split('\n\n') if len(p.strip()) >= min_chars]
        if paras:
            return paras[:max_count]

        # Fallback: when paragraph separators are missing, use long sentence windows.
        flattened = DeepSemanticEngine._normalize_text(text)
        if not flattened:
            return []

        fallback = []
        pattern = r'[^。！？；]+[。！？；]?'
        import re

        chunks = [s.strip() for s in re.findall(pattern, flattened) if s.strip()]
        current = []
        current_len = 0
        for sent in chunks:
            current.append(sent)
            current_len += len(sent)
            if current_len >= min_chars:
                fallback.append(''.join(current))
                current = []
                current_len = 0
            if len(fallback) >= max_count:
                break

        if current and len(''.join(current)) >= min_chars and len(fallback) < max_count:
            fallback.append(''.join(current))
        return fallback

    @staticmethod
    def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not intervals:
            return []

        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]
        for start, end in sorted_intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _sum_intervals(intervals: List[Tuple[int, int]]) -> int:
        if not intervals:
            return 0
        return int(sum(max(0, end - start) for start, end in intervals))

    def _collect_target_intervals(self, plagiarized_parts: List[Dict], target_len: int) -> List[Tuple[int, int]]:
        if target_len <= 0 or not plagiarized_parts:
            return []

        intervals = []
        for part in plagiarized_parts:
            start = part.get('target_start')
            end = part.get('target_end')
            if isinstance(start, int) and isinstance(end, int) and end > start:
                s = max(0, start)
                e = min(target_len, end)
                if e > s:
                    intervals.append((s, e))
        return intervals

    def _calculate_raw_coverage(self, plagiarized_parts: List[Dict], target_len: int) -> float:
        intervals = self._collect_target_intervals(plagiarized_parts, target_len)
        if not intervals:
            return 0.0

        merged = self._merge_intervals(intervals)
        return self._clamp01(self._sum_intervals(merged) / max(1, target_len))

    def _calculate_coverage(self, plagiarized_parts: List[Dict], target_len: int) -> float:
        if target_len <= 0 or not plagiarized_parts:
            return 0.0

        # Step5: coverage is confidence-weighted interval union,
        # so downgraded matches contribute less instead of hard removal.
        events = []
        for part in plagiarized_parts:
            start = part.get('target_start')
            end = part.get('target_end')
            if isinstance(start, int) and isinstance(end, int) and end > start:
                s = max(0, start)
                e = min(target_len, end)
                if e <= s:
                    continue
                conf = self._clamp01(float(part.get('confidence', part.get('score', 0.0))))
                # Keep very-low confidence hits lower impact, but avoid over-suppressing balanced mode.
                conf = conf ** 1.10
                events.append((s, 1, round(conf, 6)))
                events.append((e, -1, round(conf, 6)))

        if not events:
            return 0.0

        import heapq
        from collections import defaultdict

        events.sort(key=lambda x: x[0])
        active_counts = defaultdict(int)
        max_heap = []

        weighted_covered = 0.0
        idx = 0
        n = len(events)

        while idx < n:
            pos = events[idx][0]
            while idx < n and events[idx][0] == pos:
                _, typ, conf = events[idx]
                if typ == 1:
                    active_counts[conf] += 1
                    heapq.heappush(max_heap, -conf)
                else:
                    active_counts[conf] -= 1
                idx += 1

            while max_heap and active_counts[-max_heap[0]] <= 0:
                heapq.heappop(max_heap)

            if idx >= n:
                break
            next_pos = events[idx][0]
            if next_pos <= pos:
                continue

            if max_heap:
                top_conf = -max_heap[0]
                weighted_covered += (next_pos - pos) * top_conf

        return self._clamp01(weighted_covered / max(1, target_len))

    def _calculate_match_confidence(self, plagiarized_parts: List[Dict]) -> float:
        if not plagiarized_parts:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0
        conf_samples = []
        for part in plagiarized_parts:
            confidence = self._clamp01(float(part.get('confidence', part.get('score', 0.0))))
            weight = float(max(1, int(part.get('length', 1))))
            weighted_sum += confidence * weight
            total_weight += weight
            conf_samples.append(confidence)

        if total_weight <= 0:
            return 0.0

        # Blend weighted mean with top-k confidence to avoid "many weak hits dilute all evidence".
        raw_conf = self._clamp01(weighted_sum / total_weight)
        if conf_samples:
            topk = min(5, len(conf_samples))
            top_mean = float(np.mean(np.sort(np.asarray(conf_samples))[-topk:]))
            raw_conf = self._clamp01(0.70 * raw_conf + 0.30 * top_mean)

        # Keep some nonlinearity for robustness, but lighter than before.
        return self._clamp01(raw_conf ** 1.35)

    def _calculate_effective_coverage(
        self,
        raw_coverage: float,
        weighted_coverage: float,
        confidence: float,
    ) -> float:
        raw_coverage = self._clamp01(raw_coverage)
        weighted_coverage = self._clamp01(weighted_coverage)
        confidence = self._clamp01(confidence)

        coverage_gap = max(0.0, raw_coverage - weighted_coverage)
        recovery = coverage_gap * (0.20 + 0.35 * confidence)
        return self._clamp01(weighted_coverage + recovery)

    def _calculate_continuity_features(
        self,
        plagiarized_parts: List[Dict],
        target_len: int,
    ) -> Dict[str, float]:
        intervals = self._collect_target_intervals(plagiarized_parts, target_len)
        if not intervals:
            return {
                "longest_run_ratio": 0.0,
                "top3_run_ratio": 0.0,
                "merged_hit_count": 0,
            }

        merged = self._merge_intervals(intervals)
        lengths = sorted((max(0, end - start) for start, end in merged), reverse=True)
        if not lengths:
            return {
                "longest_run_ratio": 0.0,
                "top3_run_ratio": 0.0,
                "merged_hit_count": 0,
            }

        longest_run_ratio = self._clamp01(lengths[0] / max(1, target_len))
        top3_run_ratio = self._clamp01(sum(lengths[:3]) / max(1, target_len))
        return {
            "longest_run_ratio": float(longest_run_ratio),
            "top3_run_ratio": float(top3_run_ratio),
            "merged_hit_count": int(len(merged)),
        }

    def _calculate_realistic_score(
        self,
        profile_name: str,
        profile_cfg: Dict,
        raw_coverage: float,
        weighted_coverage: float,
        confidence: float,
        doc_semantic: float,
        paragraph_semantic: float,
        hit_count: int,
        longest_run_ratio: float,
        top3_run_ratio: float,
    ) -> Tuple[float, float, float, float, float, float]:
        effective_coverage = self._calculate_effective_coverage(
            raw_coverage,
            weighted_coverage,
            confidence,
        )

        final_cfg = profile_cfg.get("final_score", {})
        semantic_weight = float(final_cfg.get("semantic_weight", 0.30))
        evidence_weight = float(final_cfg.get("evidence_weight", 0.70))
        semantic_center = float(final_cfg.get("semantic_center", 0.80))
        semantic_scale = max(1e-6, float(final_cfg.get("semantic_scale", 0.08)))
        coverage_gain = max(1e-6, float(final_cfg.get("coverage_gain", 8.0)))
        low_evidence_cap_base = float(final_cfg.get("low_evidence_cap_base", 0.14))
        low_evidence_cap_gain = float(final_cfg.get("low_evidence_cap_gain", 0.20))
        continuity_boost = float(final_cfg.get("continuity_boost", 0.06))

        semantic_input = self._clamp01(0.60 * doc_semantic + 0.40 * paragraph_semantic)
        semantic_score = self._sigmoid((semantic_input - semantic_center) / semantic_scale)

        coverage_core = self._clamp01(
            0.55 * effective_coverage
            + 0.25 * longest_run_ratio
            + 0.20 * top3_run_ratio
        )
        evidence_score = self._clamp01(1.0 - math.exp(-coverage_gain * coverage_core))
        confidence_scale = 0.72 + 0.28 * (self._clamp01(confidence) ** 0.85)
        evidence_score = self._clamp01(evidence_score * confidence_scale)

        continuity_signal = self._clamp01(
            max(longest_run_ratio * 10.0, top3_run_ratio * 5.0)
        )
        continuity_bonus = continuity_boost * continuity_signal * self._clamp01(confidence)

        final_score = (
            semantic_weight * semantic_score
            + evidence_weight * evidence_score
            + continuity_bonus
        )

        if effective_coverage < 0.03 and longest_run_ratio < 0.01:
            low_evidence_cap = low_evidence_cap_base + low_evidence_cap_gain * semantic_score
            final_score = min(final_score, low_evidence_cap)
        else:
            low_evidence_cap = 1.0

        return (
            self._clamp01(final_score),
            float(effective_coverage),
            float(semantic_score),
            float(evidence_score),
            float(continuity_bonus),
            float(low_evidence_cap),
        )

    def _get_target_semantic_context(self, target_text: str):
        target_norm = self._normalize_text(target_text)

        if target_norm == self._cache_target_doc_text:
            return (
                self._cache_target_doc_embedding,
                self._cache_target_doc_paragraphs,
                self._cache_target_para_embeddings,
            )

        target_vec = self.encode([target_norm])[0] if target_norm else None
        target_paras = self._get_paragraphs(target_text, min_chars=80)
        target_para_emb = self.encode(target_paras) if target_paras else None

        self._cache_target_doc_text = target_norm
        self._cache_target_doc_embedding = target_vec
        self._cache_target_doc_paragraphs = target_paras
        self._cache_target_para_embeddings = target_para_emb

        return target_vec, target_paras, target_para_emb

    def _calculate_document_semantic(self, target_text: str, ref_text: str) -> Dict[str, float]:
        target_vec, target_paras, target_para_emb = self._get_target_semantic_context(target_text)

        ref_norm = self._normalize_text(ref_text)
        if target_vec is None or not ref_norm:
            return {
                "doc_semantic": 0.0,
                "global_semantic": 0.0,
                "paragraph_semantic": 0.0,
            }

        ref_vec = self.encode([ref_norm])[0]
        global_sim = self._clamp01(float(np.dot(target_vec, ref_vec)))

        paragraph_semantic = global_sim
        ref_paras = self._get_paragraphs(ref_text, min_chars=80)
        if target_para_emb is not None and ref_paras:
            ref_para_emb = self.encode(ref_paras)
            if ref_para_emb.size > 0 and target_para_emb.size > 0:
                sim_matrix = np.dot(target_para_emb, ref_para_emb.T)
                best_per_target = np.max(sim_matrix, axis=1)
                if best_per_target.size > 0:
                    topk = min(3, best_per_target.size)
                    topk_mean = float(np.mean(np.sort(best_per_target)[-topk:]))
                    overall_mean = float(np.mean(best_per_target))
                    paragraph_semantic = self._clamp01(0.65 * topk_mean + 0.35 * overall_mean)

        doc_semantic = self._clamp01(0.65 * global_sim + 0.35 * paragraph_semantic)
        return {
            "doc_semantic": doc_semantic,
            "global_semantic": global_sim,
            "paragraph_semantic": paragraph_semantic,
        }

    def score_document_pair(
        self,
        target_text: str,
        ref_text: str,
        plagiarized_parts: Optional[List[Dict]] = None,
        threshold_profile: str = "balanced",
    ) -> Dict[str, float]:
        """
        Final score is calibrated as a practical composite similarity score:
        document-level semantics participate directly, while local coverage,
        confidence, and continuity provide the concrete evidence strength.
        The older gate-heavy score is retained separately as risk_score.
        """
        profile_name, profile_cfg = self._resolve_profile(threshold_profile)

        if plagiarized_parts is None:
            plagiarized_parts = self.sliding_window_check(
                target_text,
                ref_text,
                threshold_profile=profile_name,
            )

        semantic_scores = self._calculate_document_semantic(target_text, ref_text)
        target_norm_len = len(self._normalize_text(target_text))
        raw_coverage = self._calculate_raw_coverage(plagiarized_parts, target_norm_len)
        weighted_coverage = self._calculate_coverage(plagiarized_parts, target_norm_len)
        confidence = self._calculate_match_confidence(plagiarized_parts)
        hit_count = int(len(plagiarized_parts))
        continuity = self._calculate_continuity_features(plagiarized_parts, target_norm_len)

        # Convert topic-level semantic similarity into plagiarism-sensitive semantic evidence.
        # Similar-domain docs can be semantically close; only the excess above profile floor contributes.
        semantic_floor = float(profile_cfg.get("semantic_floor", 0.0))
        semantic_excess = self._clamp01(
            (semantic_scores["doc_semantic"] - semantic_floor) / max(1e-6, 1.0 - semantic_floor)
        )

        final_score, effective_coverage, semantic_signal, evidence_score, continuity_bonus, low_evidence_cap = self._calculate_realistic_score(
            profile_name,
            profile_cfg,
            raw_coverage,
            weighted_coverage,
            confidence,
            semantic_scores["doc_semantic"],
            semantic_scores["paragraph_semantic"],
            hit_count,
            continuity["longest_run_ratio"],
            continuity["top3_run_ratio"],
        )

        weights = profile_cfg["score_weights"]
        base_score = (
            weights["doc_semantic"] * semantic_excess
            + weights["coverage"] * weighted_coverage
            + weights["confidence"] * confidence
        )

        gate_cfg = profile_cfg["score_gate"]
        evidence_strength = (weighted_coverage * confidence) ** 0.5
        gate = self._clamp01(
            gate_cfg["base"]
            + gate_cfg["coverage"] * weighted_coverage
            + gate_cfg["confidence"] * confidence
            + gate_cfg.get("evidence", 0.0) * evidence_strength
        )

        risk_score = self._clamp01(base_score * gate)

        # Strong suppression for "same topic, low concrete overlap" cases.
        if weighted_coverage < gate_cfg["low_cov_th"]:
            low_cov_cap = gate_cfg.get("low_cov_cap")
            if low_cov_cap is not None:
                risk_score = min(risk_score, low_cov_cap)
        elif weighted_coverage < gate_cfg["mid_cov_th"] and confidence < gate_cfg["mid_conf_th"]:
            risk_score = min(risk_score, gate_cfg["mid_cov_cap"])
        elif (
            weighted_coverage < gate_cfg.get("topic_cov_th", 0.0)
            and confidence < gate_cfg.get("topic_conf_th", 1.0)
        ):
            risk_score = min(risk_score, gate_cfg.get("topic_cap", 1.0))

        # Additional guard for same-domain cases: moderate confidence but weak coverage.
        if (
            weighted_coverage < gate_cfg.get("low_evidence_cov_th", 0.0)
            and confidence < gate_cfg.get("low_evidence_conf_th", 1.0)
        ):
            risk_score = min(risk_score, gate_cfg.get("low_evidence_cap", 1.0))

        return {
            "profile": profile_name,
            "doc_semantic": float(semantic_scores["doc_semantic"]),
            "doc_semantic_excess": float(semantic_excess),
            "doc_semantic_global": float(semantic_scores["global_semantic"]),
            "doc_semantic_paragraph": float(semantic_scores["paragraph_semantic"]),
            "coverage": float(raw_coverage),
            "coverage_raw": float(raw_coverage),
            "coverage_weighted": float(weighted_coverage),
            "coverage_effective": float(effective_coverage),
            "confidence": float(confidence),
            "continuity_longest": float(continuity["longest_run_ratio"]),
            "continuity_top3": float(continuity["top3_run_ratio"]),
            "continuity_hit_groups": int(continuity["merged_hit_count"]),
            "base_score": float(self._clamp01(base_score)),
            "gate": float(gate),
            "final_score": float(self._clamp01(final_score)),
            "risk_score": float(self._clamp01(risk_score)),
            "semantic_signal": float(semantic_signal),
            "evidence_score": float(evidence_score),
            "continuity_bonus": float(continuity_bonus),
            "low_evidence_cap": float(low_evidence_cap),
            "hit_count": hit_count,
        }

    def sliding_window_check(self, target_text, ref_text, window_size=50, threshold_profile="balanced"):
        """Fine-grained semantic matching with dynamic thresholds and anti-false-positive rules."""
        import torch

        resolved_profile_name, profile_cfg = self._resolve_profile(threshold_profile)
        print(f">>> [BGE] Threshold profile: {resolved_profile_name}")

        target_norm = self._normalize_text(target_text)
        ref_norm = self._normalize_text(ref_text)

        if self._cache_target_text == target_norm:
            win1 = self._cache_win1
            emb1 = self._cache_emb1
        else:
            win1 = self._build_windows(target_norm)
            if not win1:
                return []
            emb1 = self.encode([w["text"] for w in win1])
            self._cache_target_text = target_norm
            self._cache_win1 = win1
            self._cache_emb1 = emb1

        win2 = self._build_windows(ref_norm)
        if not win2:
            return []

        emb2 = self.encode([w["text"] for w in win2])
        if emb1.size == 0 or emb2.size == 0:
            return []

        results = []
        short_text_max_chars = profile_cfg["short_text_max_chars"]
        is_super_short = len(target_norm) < short_text_max_chars and len(ref_norm) < short_text_max_chars
        sim_matrix = np.dot(emb1, emb2.T)
        topk = min(self.window_topk, sim_matrix.shape[1])

        for i, item1 in enumerate(win1):
            sims = sim_matrix[i]
            candidate_indices = self._select_topk_indices(sims, topk)
            if candidate_indices.size == 0:
                continue

            best_idx = int(candidate_indices[0])
            max_sim = float(sims[best_idx])
            outlier_metrics = self._resolve_outlier_metrics(
                sims,
                peak_sim=max_sim,
                profile_cfg=profile_cfg,
            )
            outlier_threshold = float(outlier_metrics["effective_threshold"])
            is_outlier_hit = bool(
                outlier_metrics["is_statistical_outlier"]
                or outlier_metrics["is_percentile_outlier"]
            )

            if is_super_short:
                threshold_passed = (
                    max_sim > profile_cfg["short_low"]
                    and (is_outlier_hit or max_sim > profile_cfg["short_high"])
                )
            else:
                threshold_passed = (
                    max_sim > profile_cfg["long_low"]
                    and (is_outlier_hit or max_sim > profile_cfg["long_high"])
                )

            if not threshold_passed:
                continue

            candidate_result = None
            candidate_low_th = profile_cfg["short_low"] if is_super_short else profile_cfg["long_low"]
            candidate_high_th = profile_cfg["short_high"] if is_super_short else profile_cfg["long_high"]
            for rank, candidate_idx in enumerate(candidate_indices):
                candidate_idx = int(candidate_idx)
                candidate_sim = float(sims[candidate_idx])
                if candidate_sim <= candidate_low_th:
                    continue
                if candidate_sim < outlier_threshold and candidate_sim <= candidate_high_th:
                    continue

                scored = self._score_window_candidate(
                    item1,
                    win2[candidate_idx],
                    raw_sim=candidate_sim,
                    outlier_threshold=outlier_threshold,
                    profile_cfg=profile_cfg,
                )
                if scored is None:
                    continue

                scored["candidate_rank"] = int(rank + 1)
                scored["outlier_threshold"] = float(outlier_threshold)
                scored["outlier_std_threshold"] = float(outlier_metrics["std_threshold"])
                scored["outlier_percentile_threshold"] = float(outlier_metrics["percentile_threshold"])
                scored["outlier_trigger"] = (
                    "percentile"
                    if outlier_metrics["is_percentile_outlier"] and not outlier_metrics["is_statistical_outlier"]
                    else "std"
                    if outlier_metrics["is_statistical_outlier"]
                    else "high"
                )
                candidate_result = scored
                break

            if candidate_result is not None:
                results.append(candidate_result)

        # Macro paragraph-level structural warning.
        target_paras = self._get_paragraphs(target_text, min_chars=50)
        ref_paras = self._get_paragraphs(ref_text, min_chars=50)

        paragraph_warnings = []
        if target_paras and ref_paras:
            para_emb1 = self.encode(target_paras)
            para_emb2 = self.encode(ref_paras)
            para_sim_matrix = np.dot(para_emb1, para_emb2.T)

            for i in range(para_sim_matrix.shape[0]):
                p_sims = para_sim_matrix[i]
                best_p_idx = int(np.argmax(p_sims))
                p_max_sim = float(p_sims[best_p_idx])

                if p_max_sim > profile_cfg["paragraph_threshold"]:
                    entities_t = self._extract_entities(target_paras[i])
                    entities_r = self._extract_entities(ref_paras[best_p_idx])
                    iou = self._safe_iou(entities_t, entities_r)

                    para_penalty = 1.0
                    if iou < 0.2:
                        para_penalty *= 0.60
                    elif iou < 0.35:
                        para_penalty *= 0.80

                    paragraph_warnings.append(
                        {
                            'target_part': "[段落结构相似] " + target_paras[i][:100] + "...",
                            'ref_part': "[段落结构相似] " + ref_paras[best_p_idx][:100] + "...",
                            'score': float(self._clamp01(p_max_sim * (0.55 + 0.45 * para_penalty))),
                            'raw_score': float(p_max_sim),
                            'confidence': float(self._clamp01(p_max_sim * (0.80 + 0.20 * iou) * para_penalty)),
                            'length': len(target_paras[i]),
                            'target_start': None,
                            'target_end': None,
                            'ref_start': None,
                            'ref_end': None,
                            'match_type': 'paragraph',
                            'rule_penalty': float(self._clamp01(para_penalty)),
                            'rule_flags': ['paragraph_entity_low_iou'] if iou < 0.35 else [],
                        }
                    )

        results.extend(paragraph_warnings)
        results.sort(key=lambda x: x['score'], reverse=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results
