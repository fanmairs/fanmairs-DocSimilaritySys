import re

import numpy as np
from typing import Dict, List, Optional, Tuple
from deep_semantic_evidence import (
    calculate_continuity_features,
    calculate_coverage,
    calculate_effective_coverage,
    calculate_match_confidence,
    calculate_raw_coverage,
    calculate_realistic_score,
    collect_target_intervals,
)
from deep_semantic_profiles import DEFAULT_PROFILE, THRESHOLD_PROFILES, resolve_profile
from deep_semantic_text import (
    clamp01,
    extract_entities,
    extract_tags,
    get_paragraphs,
    get_skeleton,
    is_formula_explanation,
    make_span,
    merge_intervals,
    normalize_for_paragraphs,
    normalize_text,
    safe_iou,
    sigmoid,
    split_sentences_with_offsets,
    sum_intervals,
)
from deep_semantic_window_scoring import (
    resolve_outlier_metrics,
    score_window_candidate,
    select_topk_indices,
)
from text_noise_filter import is_numeric_table_noise


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
        self.default_profile = DEFAULT_PROFILE
        self.threshold_profiles = THRESHOLD_PROFILES

    @staticmethod
    def _clamp01(value: float) -> float:
        return clamp01(value)

    @staticmethod
    def _sigmoid(value: float) -> float:
        return sigmoid(value)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return normalize_text(text)

    @staticmethod
    def _normalize_for_paragraphs(text: str) -> str:
        return normalize_for_paragraphs(text)

    def _resolve_profile(self, profile_name: Optional[str]) -> Tuple[str, Dict]:
        return resolve_profile(profile_name, self.default_profile, self.threshold_profiles)

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
        return make_span(text, start, end)

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
        return split_sentences_with_offsets(text)

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
            if is_numeric_table_noise(window["text"]):
                continue
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', window["text"]))
            if len(window["text"]) <= 5:
                continue
            if chinese_chars / max(len(window["text"]), 1) <= 0.08:
                continue
            if chinese_chars > 0 and chinese_chars / max(len(window["text"]), 1) <= 0.2:
                continue
            filtered.append(window)

        return filtered or windows

    @staticmethod
    def _select_topk_indices(scores: np.ndarray, topk: int) -> np.ndarray:
        return select_topk_indices(scores, topk)

    def _resolve_outlier_metrics(self, sims: np.ndarray, peak_sim: float, profile_cfg: Dict) -> Dict[str, float]:
        return resolve_outlier_metrics(self, sims, peak_sim, profile_cfg)

    def _score_window_candidate(
        self,
        item1: Dict,
        item2: Dict,
        raw_sim: float,
        outlier_threshold: float,
        profile_cfg: Dict,
    ) -> Optional[Dict]:
        return score_window_candidate(self, item1, item2, raw_sim, outlier_threshold, profile_cfg)

    @staticmethod
    def _extract_entities(text: str):
        return extract_entities(text)

    @staticmethod
    def _extract_tags(text: str):
        return extract_tags(text)

    @staticmethod
    def _get_skeleton(text: str) -> str:
        return get_skeleton(text)

    @staticmethod
    def _is_formula_explanation(text: str) -> bool:
        return is_formula_explanation(text)

    @staticmethod
    def _safe_iou(set_a: set, set_b: set, default: float = 1.0) -> float:
        return safe_iou(set_a, set_b, default)

    @staticmethod
    def _get_paragraphs(text: str, min_chars: int = 50, max_count: int = 24) -> List[str]:
        return get_paragraphs(text, min_chars, max_count)

    @staticmethod
    def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        return merge_intervals(intervals)

    @staticmethod
    def _sum_intervals(intervals: List[Tuple[int, int]]) -> int:
        return sum_intervals(intervals)

    def _collect_target_intervals(self, plagiarized_parts: List[Dict], target_len: int) -> List[Tuple[int, int]]:
        return collect_target_intervals(self, plagiarized_parts, target_len)

    def _calculate_raw_coverage(self, plagiarized_parts: List[Dict], target_len: int) -> float:
        return calculate_raw_coverage(self, plagiarized_parts, target_len)

    def _calculate_coverage(self, plagiarized_parts: List[Dict], target_len: int) -> float:
        return calculate_coverage(self, plagiarized_parts, target_len)

    def _calculate_match_confidence(self, plagiarized_parts: List[Dict]) -> float:
        return calculate_match_confidence(self, plagiarized_parts)

    def _calculate_effective_coverage(
        self,
        raw_coverage: float,
        weighted_coverage: float,
        confidence: float,
    ) -> float:
        return calculate_effective_coverage(self, raw_coverage, weighted_coverage, confidence)

    def _calculate_continuity_features(
        self,
        plagiarized_parts: List[Dict],
        target_len: int,
    ) -> Dict[str, float]:
        return calculate_continuity_features(self, plagiarized_parts, target_len)

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
        return calculate_realistic_score(
            self,
            profile_name,
            profile_cfg,
            raw_coverage,
            weighted_coverage,
            confidence,
            doc_semantic,
            paragraph_semantic,
            hit_count,
            longest_run_ratio,
            top3_run_ratio,
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
