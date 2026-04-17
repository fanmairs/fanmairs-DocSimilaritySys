from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from api_bge_helpers import (
    BGE_STRATEGY_COARSE,
    build_basic_bert_result,
    run_bert_fine_verification,
)
from evidence import GlobalEvidenceAggregator
from reports import build_report_payload, build_traditional_result, sort_report_items


class TaskRunner:
    """Owns heavyweight engines and computes one queued task at a time."""

    def __init__(self):
        self.bert_engine = None
        self.traditional_system = None
        self.coarse_retriever = None
        self.global_evidence_aggregator = None
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return

        from engines.semantic.bge_backend import DeepSemanticEngine
        from engines.semantic.coarse_retrieval import CoarseRetriever
        from engines.traditional.system import PlagiarismDetectorSystem

        print(">>> [GPU Queue] Loading semantic and traditional engines...")
        self.bert_engine = DeepSemanticEngine()
        self.traditional_system = PlagiarismDetectorSystem(
            stopwords_path="dicts/stopwords.txt",
            lsa_components=3,
            synonyms_path="dicts/synonyms.txt",
            semantic_embeddings_path="dicts/embeddings/fasttext_zh.vec",
            semantic_threshold=0.55,
            semantic_weight=0.35,
        )
        self.coarse_retriever = CoarseRetriever(
            self.bert_engine,
            self.traditional_system.preprocessor,
        )
        self.global_evidence_aggregator = GlobalEvidenceAggregator(self.bert_engine)
        self.loaded = True
        print(">>> [GPU Queue] Engines loaded. Worker is ready.")

    def is_ready(self) -> bool:
        return bool(
            self.loaded
            and self.bert_engine is not None
            and self.traditional_system is not None
        )

    def read_document(self, path: str, *, body_mode: bool = False) -> str:
        self.load()
        text = self.traditional_system.read_document(path)
        if body_mode:
            text = self.traditional_system.clean_academic_noise(text)
        return text

    def estimate_window_count(self, text: str) -> int:
        self.load()
        from engines.semantic.bge_backend import DeepSemanticEngine

        normalized = DeepSemanticEngine._normalize_text(text)
        if not normalized:
            return 0
        return len(self.bert_engine._build_windows(normalized))

    def process(self, task: Dict[str, object]) -> Dict[str, object]:
        self.load()
        mode = str(task.get("mode", "bert"))
        body_mode = bool(task.get("body_mode", False))
        target_path = str(task["target_path"])
        ref_paths = [str(path) for path in task.get("ref_paths", [])]

        if mode == "bert":
            return self._process_semantic_task(
                target_path=target_path,
                ref_paths=ref_paths,
                body_mode=body_mode,
                bert_profile=str(task.get("bert_profile", "balanced")),
                bge_strategy=str(task.get("bge_strategy", BGE_STRATEGY_COARSE)),
                coarse_config=task.get("coarse_config"),
            )

        return self._process_traditional_task(
            target_path=target_path,
            ref_paths=ref_paths,
            body_mode=body_mode,
        )

    def _load_reference_payloads(
        self,
        ref_paths: Sequence[str],
        *,
        body_mode: bool,
    ) -> tuple[List[Dict[str, str]], Dict[str, str]]:
        reference_payloads: List[Dict[str, str]] = []
        reference_text_map: Dict[str, str] = {}
        for ref_path in ref_paths:
            ref_text = self.read_document(ref_path, body_mode=body_mode)
            reference_payloads.append({"path": ref_path, "text": ref_text})
            reference_text_map[ref_path] = ref_text
        return reference_payloads, reference_text_map

    def _process_semantic_task(
        self,
        *,
        target_path: str,
        ref_paths: Sequence[str],
        body_mode: bool,
        bert_profile: str,
        bge_strategy: str,
        coarse_config: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        target_text = self.read_document(target_path, body_mode=body_mode)
        reference_payloads, reference_text_map = self._load_reference_payloads(
            ref_paths,
            body_mode=body_mode,
        )

        results: List[Dict[str, object]] = []
        verified_results: List[Dict[str, object]] = []
        coarse_only_results: List[Dict[str, object]] = []
        candidate_count = len(reference_payloads)

        print(
            ">>> [BGE][Strategy] "
            f"strategy={bge_strategy} references={len(reference_payloads)}"
        )

        if bge_strategy == BGE_STRATEGY_COARSE:
            task_coarse_retriever = self.coarse_retriever.with_config(coarse_config)
            target_context = task_coarse_retriever.build_target_context(target_text)
            reference_contexts = task_coarse_retriever.build_reference_contexts(reference_payloads)
            ranked_refs, selection_meta = task_coarse_retriever.rank_references(
                target_context,
                reference_contexts,
            )
            ranked_ref_map = {item["path"]: item for item in ranked_refs}
            candidate_ref_paths = [
                item["path"]
                for item in ranked_refs
                if item.get("is_candidate", False)
            ]
            coarse_only_results = [
                task_coarse_retriever.build_coarse_only_result(item, bert_profile)
                for item in ranked_refs
                if not item.get("is_candidate", False)
            ]
            for item in coarse_only_results:
                item["retrieval_strategy"] = bge_strategy

            candidate_count = int(selection_meta.get("candidate_count", len(candidate_ref_paths)))
            print(
                ">>> [BGE][Coarse] "
                f"references={len(reference_payloads)} "
                f"candidates={selection_meta['candidate_count']} "
                f"candidate_limit={selection_meta['candidate_limit']} "
                f"topic_concentrated={selection_meta['topic_concentrated']} "
                f"theme_mean={selection_meta['theme_mean']:.4f} "
                f"theme_std={selection_meta['theme_std']:.4f}"
            )

            for ref_path in candidate_ref_paths:
                ref_text = reference_text_map[ref_path]
                plagiarized_parts, score_breakdown = run_bert_fine_verification(
                    self.bert_engine,
                    ref_path,
                    target_text,
                    ref_text,
                    bert_profile,
                )
                verified_result = build_basic_bert_result(
                    ref_path,
                    bert_profile,
                    score_breakdown,
                    plagiarized_parts,
                )
                verified_result.update(
                    task_coarse_retriever.build_verified_result(
                        ranked_ref_map[ref_path],
                        bert_profile,
                        score_breakdown,
                        plagiarized_parts,
                    )
                )
                verified_result["retrieval_strategy"] = bge_strategy
                results.append(verified_result)
                verified_results.append(verified_result)
        else:
            candidate_ref_paths = list(ref_paths)
            candidate_count = len(candidate_ref_paths)
            print(
                ">>> [BGE][FullFine] "
                f"references={len(reference_payloads)} "
                "candidates=all"
            )

            for index, ref_path in enumerate(candidate_ref_paths, start=1):
                ref_text = reference_text_map[ref_path]
                plagiarized_parts, score_breakdown = run_bert_fine_verification(
                    self.bert_engine,
                    ref_path,
                    target_text,
                    ref_text,
                    bert_profile,
                )
                verified_result = build_basic_bert_result(
                    ref_path,
                    bert_profile,
                    score_breakdown,
                    plagiarized_parts,
                )
                verified_result.update(
                    {
                        "sim_bert_candidate_rank": index,
                        "sim_bert_coarse_rank": None,
                        "retrieval_strategy": bge_strategy,
                        "retrieval_reason": "full_fine",
                        "retrieval_candidate_pool_size": len(candidate_ref_paths),
                        "retrieval_reference_count": len(reference_payloads),
                        "retrieval_theme_mean": 0.0,
                        "retrieval_theme_std": 0.0,
                        "retrieval_topic_concentrated": False,
                    }
                )
                results.append(verified_result)
                verified_results.append(verified_result)

        result_summary = self.global_evidence_aggregator.aggregate(
            target_text,
            verified_results,
            bert_profile=bert_profile,
            reference_count=len(reference_payloads),
            candidate_count=candidate_count,
            retrieval_strategy=bge_strategy,
        )
        print(
            ">>> [BGE][Global] "
            f"score={result_summary['global_score']:.4f} "
            f"coverage={result_summary['global_coverage_effective']:.4f} "
            f"confidence={result_summary['global_confidence']:.4f} "
            f"source_diversity={result_summary['global_source_diversity']:.4f} "
            f"verified_sources={result_summary['global_verified_source_count']}"
        )

        results.extend(coarse_only_results)
        results = sort_report_items(results, mode="bert")
        return build_report_payload(results, result_summary)

    def _process_traditional_task(
        self,
        *,
        target_path: str,
        ref_paths: Sequence[str],
        body_mode: bool,
    ) -> Dict[str, object]:
        raw_results = self.traditional_system.check_similarity(
            target_path,
            list(ref_paths),
            body_mode=body_mode,
        )
        results = sort_report_items(
            [build_traditional_result(item) for item in raw_results],
            mode="traditional",
        )
        return build_report_payload(results, None)


TaskProcessor = TaskRunner
