from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from deep_semantic import DeepSemanticEngine


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass
class CoarseRetrievalConfig:
    min_candidates: int = 8
    max_candidates: int = 20
    base_candidate_ratio: float = 0.15

    concentrated_min_candidates: int = 12
    concentrated_max_candidates: int = 40
    concentrated_candidate_ratio: float = 0.30

    coarse_threshold: float = 0.58
    lexical_threshold: float = 0.24
    paragraph_hotspot_threshold: float = 0.80
    per_paragraph_top_m: int = 1

    paragraph_min_chars: int = 80
    paragraph_max_count: int = 14
    paragraph_score_top_k: int = 3
    lexical_top_terms: int = 96

    topic_mean_threshold: float = 0.80
    topic_std_threshold: float = 0.03

    def to_dict(self) -> Dict[str, float]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
        }

    def normalized(self) -> "CoarseRetrievalConfig":
        normalized_values = self.to_dict()

        int_fields = {
            "min_candidates",
            "max_candidates",
            "concentrated_min_candidates",
            "concentrated_max_candidates",
            "per_paragraph_top_m",
            "paragraph_min_chars",
            "paragraph_max_count",
            "paragraph_score_top_k",
            "lexical_top_terms",
        }
        ratio_fields = {
            "base_candidate_ratio",
            "concentrated_candidate_ratio",
            "coarse_threshold",
            "lexical_threshold",
            "paragraph_hotspot_threshold",
            "topic_mean_threshold",
            "topic_std_threshold",
        }

        for name in int_fields:
            normalized_values[name] = max(1, int(round(normalized_values[name])))

        for name in ratio_fields:
            normalized_values[name] = _clamp01(float(normalized_values[name]))

        normalized_values["max_candidates"] = max(
            normalized_values["min_candidates"],
            normalized_values["max_candidates"],
        )
        normalized_values["concentrated_min_candidates"] = max(
            normalized_values["min_candidates"],
            normalized_values["concentrated_min_candidates"],
        )
        normalized_values["concentrated_max_candidates"] = max(
            normalized_values["concentrated_min_candidates"],
            normalized_values["concentrated_max_candidates"],
        )
        normalized_values["paragraph_score_top_k"] = min(
            normalized_values["paragraph_score_top_k"],
            normalized_values["paragraph_max_count"],
        )

        return CoarseRetrievalConfig(**normalized_values)

    @classmethod
    def from_partial_dict(
        cls,
        raw_config: Optional[Dict[str, object]] = None,
        *,
        base: Optional["CoarseRetrievalConfig"] = None,
    ) -> "CoarseRetrievalConfig":
        base_config = (base or cls()).normalized()
        if not raw_config:
            return base_config

        merged_values = base_config.to_dict()
        default_values = cls().to_dict()

        for key, raw_value in raw_config.items():
            if key not in merged_values or raw_value in (None, ""):
                continue

            default_value = default_values[key]
            try:
                if isinstance(default_value, int) and not isinstance(default_value, bool):
                    merged_values[key] = int(float(raw_value))
                else:
                    merged_values[key] = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid coarse retrieval value for '{key}'") from exc

        return cls(**merged_values).normalized()


@dataclass
class TargetContext:
    raw_text: str
    normalized_text: str
    doc_embedding: Optional[np.ndarray]
    paragraphs: List[str]
    paragraph_embeddings: np.ndarray
    token_weights: Counter = field(default_factory=Counter)


@dataclass
class ReferenceContext:
    index: int
    path: str
    file: str
    raw_text: str
    normalized_text: str
    doc_embedding: Optional[np.ndarray]
    paragraphs: List[str]
    paragraph_embeddings: np.ndarray
    token_weights: Counter = field(default_factory=Counter)


def compute_candidate_limit(
    total_refs: int,
    config: Optional[CoarseRetrievalConfig] = None,
    *,
    topic_concentrated: bool = False,
) -> int:
    cfg = config or CoarseRetrievalConfig()
    if total_refs <= 0:
        return 0

    if topic_concentrated:
        scaled = int(math.ceil(total_refs * cfg.concentrated_candidate_ratio))
        scaled = max(cfg.concentrated_min_candidates, scaled)
        scaled = min(cfg.concentrated_max_candidates, scaled)
        return max(1, min(total_refs, scaled))

    scaled = int(math.ceil(total_refs * cfg.base_candidate_ratio))
    scaled = max(cfg.min_candidates, scaled)
    scaled = min(cfg.max_candidates, scaled)
    return max(1, min(total_refs, scaled))


def analyze_topic_concentration(
    scored_refs: Sequence[Dict],
    config: Optional[CoarseRetrievalConfig] = None,
) -> Dict[str, float]:
    cfg = config or CoarseRetrievalConfig()
    if not scored_refs:
        return {"mean": 0.0, "std": 0.0, "is_concentrated": False}

    top_doc_scores = sorted(
        (float(item.get("doc_semantic", 0.0)) for item in scored_refs),
        reverse=True,
    )[: min(20, len(scored_refs))]
    if not top_doc_scores:
        return {"mean": 0.0, "std": 0.0, "is_concentrated": False}

    mean_score = float(np.mean(top_doc_scores))
    std_score = float(np.std(top_doc_scores))
    is_concentrated = (
        len(top_doc_scores) >= 6
        and mean_score >= cfg.topic_mean_threshold
        and std_score <= cfg.topic_std_threshold
    )
    return {"mean": mean_score, "std": std_score, "is_concentrated": is_concentrated}


def select_candidate_indices(
    scored_refs: Sequence[Dict],
    *,
    target_paragraph_count: int,
    config: Optional[CoarseRetrievalConfig] = None,
    theme_stats: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    cfg = config or CoarseRetrievalConfig()
    theme_stats = theme_stats or analyze_topic_concentration(scored_refs, cfg)
    topic_concentrated = bool(theme_stats.get("is_concentrated", False))
    candidate_limit = compute_candidate_limit(
        len(scored_refs),
        cfg,
        topic_concentrated=topic_concentrated,
    )

    selected = set()
    reasons = defaultdict(list)

    ranked_indices = sorted(
        range(len(scored_refs)),
        key=lambda idx: float(scored_refs[idx].get("coarse_score", 0.0)),
        reverse=True,
    )

    for idx in ranked_indices[:candidate_limit]:
        selected.add(idx)
        reasons[idx].append("topk")

    for idx, item in enumerate(scored_refs):
        if float(item.get("coarse_score", 0.0)) >= cfg.coarse_threshold:
            selected.add(idx)
            reasons[idx].append("coarse_threshold")

        if float(item.get("lexical_anchor", 0.0)) >= cfg.lexical_threshold:
            selected.add(idx)
            reasons[idx].append("lexical_anchor")

    for para_idx in range(target_paragraph_count):
        ranked_by_hotspot = sorted(
            range(len(scored_refs)),
            key=lambda idx: float(
                (
                    scored_refs[idx].get("best_target_paragraph_scores", [])
                    or [0.0] * target_paragraph_count
                )[para_idx]
                if para_idx < len(scored_refs[idx].get("best_target_paragraph_scores", []))
                else 0.0
            ),
            reverse=True,
        )

        hits = 0
        for idx in ranked_by_hotspot:
            paragraph_scores = scored_refs[idx].get("best_target_paragraph_scores", [])
            hotspot_score = (
                float(paragraph_scores[para_idx])
                if para_idx < len(paragraph_scores)
                else 0.0
            )
            if hotspot_score < cfg.paragraph_hotspot_threshold:
                if hits > 0:
                    break
                continue

            selected.add(idx)
            reasons[idx].append(f"paragraph_hotspot_{para_idx + 1}")
            hits += 1
            if hits >= cfg.per_paragraph_top_m:
                break

    candidate_indices = sorted(
        selected,
        key=lambda idx: float(scored_refs[idx].get("coarse_score", 0.0)),
        reverse=True,
    )
    candidate_ranks = {idx: rank for rank, idx in enumerate(candidate_indices, start=1)}

    return {
        "candidate_indices": candidate_indices,
        "candidate_reasons": {
            idx: ",".join(dict.fromkeys(reasons[idx]))
            for idx in candidate_indices
        },
        "candidate_ranks": candidate_ranks,
        "candidate_limit": candidate_limit,
        "candidate_count": len(candidate_indices),
        "theme_mean": float(theme_stats.get("mean", 0.0)),
        "theme_std": float(theme_stats.get("std", 0.0)),
        "topic_concentrated": topic_concentrated,
    }


class CoarseRetriever:
    def __init__(
        self,
        semantic_engine: DeepSemanticEngine,
        text_preprocessor,
        config: Optional[CoarseRetrievalConfig] = None,
    ):
        self.semantic_engine = semantic_engine
        self.text_preprocessor = text_preprocessor
        self.config = (config or CoarseRetrievalConfig()).normalized()

    def with_config(
        self,
        config_override: Optional[Dict[str, object]] = None,
    ) -> "CoarseRetriever":
        if not config_override:
            return self
        resolved_config = CoarseRetrievalConfig.from_partial_dict(
            config_override,
            base=self.config,
        )
        return CoarseRetriever(
            self.semantic_engine,
            self.text_preprocessor,
            resolved_config,
        )

    def _encode_texts(self, texts: Sequence[str]) -> List[Optional[np.ndarray]]:
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        non_empty_texts = []
        non_empty_indices = []
        for idx, text in enumerate(texts):
            normalized = (text or "").strip()
            if not normalized:
                continue
            non_empty_texts.append(normalized)
            non_empty_indices.append(idx)

        if not non_empty_texts:
            return results

        encoded = self.semantic_engine.encode(non_empty_texts)
        for idx, emb in zip(non_empty_indices, encoded):
            results[idx] = emb
        return results

    def _token_weights(self, text: str) -> Counter:
        tokens = self.text_preprocessor.clean_and_cut(text or "")
        counter = Counter(tokens)
        if len(counter) <= self.config.lexical_top_terms:
            return counter
        return Counter(dict(counter.most_common(self.config.lexical_top_terms)))

    @staticmethod
    def _weighted_jaccard(a_weights: Counter, b_weights: Counter) -> float:
        if not a_weights or not b_weights:
            return 0.0

        keys = set(a_weights) | set(b_weights)
        numerator = 0.0
        denominator = 0.0
        for key in keys:
            left = float(a_weights.get(key, 0.0))
            right = float(b_weights.get(key, 0.0))
            numerator += min(left, right)
            denominator += max(left, right)
        if denominator <= 0:
            return 0.0
        return _clamp01(numerator / denominator)

    def _compose_coarse_score(
        self,
        doc_semantic: float,
        paragraph_hotspot: float,
        lexical_anchor: float,
    ) -> float:
        score = 0.45 * paragraph_hotspot + 0.35 * doc_semantic + 0.20 * lexical_anchor

        # Suppress same-topic false positives when semantic similarity lacks lexical anchors.
        if doc_semantic >= 0.86 and paragraph_hotspot < 0.58 and lexical_anchor < 0.12:
            score = min(score, 0.62)

        # Allow paragraph hotspots to dominate when local evidence is clearly strong.
        if paragraph_hotspot >= 0.82:
            score = max(score, 0.55 * paragraph_hotspot + 0.25 * doc_semantic + 0.20 * lexical_anchor)

        return _clamp01(score)

    def build_target_context(self, text: str) -> TargetContext:
        normalized_text = DeepSemanticEngine._normalize_text(text)
        doc_embedding = self._encode_texts([normalized_text])[0]
        paragraphs = DeepSemanticEngine._get_paragraphs(
            text,
            min_chars=self.config.paragraph_min_chars,
            max_count=self.config.paragraph_max_count,
        )
        paragraph_embeddings = self.semantic_engine.encode(paragraphs) if paragraphs else np.array([])
        token_weights = self._token_weights(text)
        return TargetContext(
            raw_text=text,
            normalized_text=normalized_text,
            doc_embedding=doc_embedding,
            paragraphs=paragraphs,
            paragraph_embeddings=paragraph_embeddings,
            token_weights=token_weights,
        )

    def build_reference_contexts(self, references: Sequence[Dict[str, str]]) -> List[ReferenceContext]:
        normalized_texts = []
        token_weights = []
        paragraphs_by_ref = []
        for ref in references:
            text = ref.get("text", "") or ""
            normalized_texts.append(DeepSemanticEngine._normalize_text(text))
            token_weights.append(self._token_weights(text))
            paragraphs_by_ref.append(
                DeepSemanticEngine._get_paragraphs(
                    text,
                    min_chars=self.config.paragraph_min_chars,
                    max_count=self.config.paragraph_max_count,
                )
            )

        doc_embeddings = self._encode_texts(normalized_texts)

        flat_paragraphs = []
        paragraph_ranges = []
        for paragraphs in paragraphs_by_ref:
            start = len(flat_paragraphs)
            flat_paragraphs.extend(paragraphs)
            paragraph_ranges.append((start, len(flat_paragraphs)))

        flat_paragraph_embeddings = (
            self.semantic_engine.encode(flat_paragraphs)
            if flat_paragraphs
            else np.array([])
        )

        contexts = []
        for idx, ref in enumerate(references):
            start, end = paragraph_ranges[idx]
            paragraph_embeddings = (
                flat_paragraph_embeddings[start:end]
                if end > start
                else np.array([])
            )
            path = ref["path"]
            contexts.append(
                ReferenceContext(
                    index=idx,
                    path=path,
                    file=os.path.basename(path).replace("ref_", ""),
                    raw_text=ref.get("text", "") or "",
                    normalized_text=normalized_texts[idx],
                    doc_embedding=doc_embeddings[idx],
                    paragraphs=paragraphs_by_ref[idx],
                    paragraph_embeddings=paragraph_embeddings,
                    token_weights=token_weights[idx],
                )
            )
        return contexts

    def _paragraph_hotspot(
        self,
        target_ctx: TargetContext,
        ref_ctx: ReferenceContext,
    ) -> Dict[str, object]:
        if (
            target_ctx.paragraph_embeddings.size == 0
            or ref_ctx.paragraph_embeddings.size == 0
        ):
            return {
                "paragraph_hotspot": 0.0,
                "best_target_paragraph_scores": [0.0] * len(target_ctx.paragraphs),
            }

        sim_matrix = np.dot(target_ctx.paragraph_embeddings, ref_ctx.paragraph_embeddings.T)
        best_per_target = np.max(sim_matrix, axis=1)
        if best_per_target.size == 0:
            return {
                "paragraph_hotspot": 0.0,
                "best_target_paragraph_scores": [0.0] * len(target_ctx.paragraphs),
            }

        topk = min(self.config.paragraph_score_top_k, best_per_target.size)
        topk_mean = float(np.mean(np.sort(best_per_target)[-topk:]))
        overall_mean = float(np.mean(best_per_target))
        paragraph_hotspot = _clamp01(0.70 * topk_mean + 0.30 * overall_mean)
        return {
            "paragraph_hotspot": paragraph_hotspot,
            "best_target_paragraph_scores": best_per_target.astype(float).tolist(),
        }

    def score_references(
        self,
        target_ctx: TargetContext,
        ref_contexts: Sequence[ReferenceContext],
    ) -> List[Dict[str, object]]:
        scored_refs = []
        for ref_ctx in ref_contexts:
            doc_semantic = 0.0
            if target_ctx.doc_embedding is not None and ref_ctx.doc_embedding is not None:
                doc_semantic = _clamp01(
                    float(np.dot(target_ctx.doc_embedding, ref_ctx.doc_embedding))
                )

            paragraph_meta = self._paragraph_hotspot(target_ctx, ref_ctx)
            lexical_anchor = self._weighted_jaccard(
                target_ctx.token_weights,
                ref_ctx.token_weights,
            )
            coarse_score = self._compose_coarse_score(
                doc_semantic=doc_semantic,
                paragraph_hotspot=float(paragraph_meta["paragraph_hotspot"]),
                lexical_anchor=lexical_anchor,
            )

            scored_refs.append(
                {
                    "index": ref_ctx.index,
                    "path": ref_ctx.path,
                    "file": ref_ctx.file,
                    "raw_text": ref_ctx.raw_text,
                    "doc_semantic": doc_semantic,
                    "paragraph_hotspot": float(paragraph_meta["paragraph_hotspot"]),
                    "lexical_anchor": lexical_anchor,
                    "coarse_score": coarse_score,
                    "best_target_paragraph_scores": paragraph_meta["best_target_paragraph_scores"],
                }
            )
        return scored_refs

    def rank_references(
        self,
        target_ctx: TargetContext,
        ref_contexts: Sequence[ReferenceContext],
    ) -> tuple[List[Dict[str, object]], Dict[str, object]]:
        scored_refs = self.score_references(target_ctx, ref_contexts)
        coarse_order = sorted(
            range(len(scored_refs)),
            key=lambda idx: float(scored_refs[idx]["coarse_score"]),
            reverse=True,
        )
        for rank, idx in enumerate(coarse_order, start=1):
            scored_refs[idx]["coarse_rank"] = rank

        selection_meta = select_candidate_indices(
            scored_refs,
            target_paragraph_count=len(target_ctx.paragraphs),
            config=self.config,
        )
        candidate_indices = set(selection_meta["candidate_indices"])
        candidate_reasons = selection_meta["candidate_reasons"]
        candidate_ranks = selection_meta["candidate_ranks"]

        for item in scored_refs:
            idx = item["index"]
            item["is_candidate"] = idx in candidate_indices
            item["candidate_reason"] = candidate_reasons.get(idx, "")
            item["candidate_rank"] = candidate_ranks.get(idx)
            item["candidate_pool_size"] = int(selection_meta["candidate_count"])
            item["reference_count"] = len(scored_refs)
            item["theme_mean"] = float(selection_meta["theme_mean"])
            item["theme_std"] = float(selection_meta["theme_std"])
            item["topic_concentrated"] = bool(selection_meta["topic_concentrated"])

        return scored_refs, selection_meta

    @staticmethod
    def build_coarse_only_result(item: Dict[str, object], bert_profile: str) -> Dict[str, object]:
        coarse_score = float(item.get("coarse_score", 0.0))
        doc_semantic = float(item.get("doc_semantic", 0.0))
        paragraph_hotspot = float(item.get("paragraph_hotspot", 0.0))
        lexical_anchor = float(item.get("lexical_anchor", 0.0))
        return {
            "file": item["file"],
            "sim_bert": coarse_score,
            "sim_bert_risk": coarse_score,
            "sim_bert_doc": doc_semantic,
            "sim_bert_doc_excess": doc_semantic,
            "sim_bert_coverage": 0.0,
            "sim_bert_coverage_raw": 0.0,
            "sim_bert_coverage_weighted": 0.0,
            "sim_bert_coverage_effective": 0.0,
            "sim_bert_confidence": 0.0,
            "sim_bert_base": coarse_score,
            "sim_bert_gate": 1.0,
            "sim_bert_hits": 0,
            "sim_bert_semantic_signal": doc_semantic,
            "sim_bert_evidence": paragraph_hotspot,
            "sim_bert_continuity_bonus": 0.0,
            "sim_bert_continuity_longest": 0.0,
            "sim_bert_continuity_top3": 0.0,
            "sim_bert_low_evidence_cap": 1.0,
            "sim_bert_legacy_coverage": 0.0,
            "sim_bert_coarse": coarse_score,
            "sim_bert_coarse_doc": doc_semantic,
            "sim_bert_coarse_para": paragraph_hotspot,
            "sim_bert_coarse_lex": lexical_anchor,
            "sim_bert_verified": False,
            "sim_bert_candidate": bool(item.get("is_candidate", False)),
            "sim_bert_candidate_rank": item.get("candidate_rank"),
            "sim_bert_coarse_rank": item.get("coarse_rank"),
            "retrieval_stage": "coarse_only",
            "retrieval_reason": item.get("candidate_reason", ""),
            "retrieval_candidate_pool_size": int(item.get("candidate_pool_size", 0)),
            "retrieval_reference_count": int(item.get("reference_count", 0)),
            "retrieval_theme_mean": float(item.get("theme_mean", 0.0)),
            "retrieval_theme_std": float(item.get("theme_std", 0.0)),
            "retrieval_topic_concentrated": bool(item.get("topic_concentrated", False)),
            "bert_profile": bert_profile,
            "plagiarized_parts": [],
        }

    @classmethod
    def build_verified_result(
        cls,
        item: Dict[str, object],
        bert_profile: str,
        score_breakdown: Dict[str, float],
        plagiarized_parts: List[Dict[str, object]],
    ) -> Dict[str, object]:
        result = cls.build_coarse_only_result(item, bert_profile)
        result.update(
            {
                "sim_bert": float(score_breakdown["final_score"]),
                "sim_bert_risk": float(
                    score_breakdown.get("risk_score", score_breakdown["final_score"])
                ),
                "sim_bert_doc": float(score_breakdown["doc_semantic"]),
                "sim_bert_doc_excess": float(
                    score_breakdown.get("doc_semantic_excess", score_breakdown["doc_semantic"])
                ),
                "sim_bert_coverage": float(score_breakdown["coverage"]),
                "sim_bert_coverage_raw": float(
                    score_breakdown.get("coverage_raw", score_breakdown["coverage"])
                ),
                "sim_bert_coverage_weighted": float(
                    score_breakdown.get("coverage_weighted", score_breakdown["coverage"])
                ),
                "sim_bert_coverage_effective": float(
                    score_breakdown.get("coverage_effective", score_breakdown["coverage"])
                ),
                "sim_bert_confidence": float(score_breakdown["confidence"]),
                "sim_bert_base": float(score_breakdown["base_score"]),
                "sim_bert_gate": float(score_breakdown["gate"]),
                "sim_bert_hits": int(score_breakdown["hit_count"]),
                "sim_bert_semantic_signal": float(
                    score_breakdown.get("semantic_signal", 0.0)
                ),
                "sim_bert_evidence": float(score_breakdown.get("evidence_score", 0.0)),
                "sim_bert_continuity_bonus": float(
                    score_breakdown.get("continuity_bonus", 0.0)
                ),
                "sim_bert_continuity_longest": float(
                    score_breakdown.get("continuity_longest", 0.0)
                ),
                "sim_bert_continuity_top3": float(
                    score_breakdown.get("continuity_top3", 0.0)
                ),
                "sim_bert_low_evidence_cap": float(
                    score_breakdown.get("low_evidence_cap", 1.0)
                ),
                "sim_bert_legacy_coverage": float(
                    score_breakdown.get("coverage_raw", score_breakdown["coverage"])
                ),
                "sim_bert_verified": True,
                "retrieval_stage": "fine_verified",
                "plagiarized_parts": plagiarized_parts,
            }
        )
        return result
