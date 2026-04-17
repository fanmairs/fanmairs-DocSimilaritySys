import unittest

import numpy as np

from scoring import (
    build_score_metadata,
    calculate_global_score,
    calculate_paragraph_hotspot,
    calculate_risk_score,
    calculate_semantic_excess,
    calculate_semantic_pair_score,
    calculate_semantic_risk_score,
    clamp01,
    compose_coarse_score,
    fuse_similarity_scores,
    resolve_outlier_metrics,
    resolve_score_level,
    select_topk_indices,
    weighted_average,
)


class ScoringCommonTests(unittest.TestCase):
    def test_common_helpers_normalize_values_and_levels(self):
        self.assertEqual(clamp01(1.5), 1.0)
        self.assertEqual(clamp01(-0.5), 0.0)
        self.assertAlmostEqual(weighted_average([0.2, 0.8], [1, 3]), 0.65)
        self.assertEqual(resolve_score_level(0.71), "high")

        metadata = build_score_metadata(
            engine="semantic",
            score=1.2,
            risk_score=0.40,
            coverage=0.30,
            confidence=0.70,
        )

        self.assertEqual(metadata["score"], 1.0)
        self.assertEqual(metadata["score_level"], "high")
        self.assertEqual(metadata["risk_level"], "medium")
        self.assertEqual(metadata["score_engine"], "semantic")


class TraditionalScoringTests(unittest.TestCase):
    def test_traditional_fusion_suppresses_lsa_only_spikes(self):
        fused = fuse_similarity_scores(
            sim_lsa=0.92,
            sim_tfidf=0.02,
            sim_soft=0.01,
            target_token_len=1000,
            ref_token_len=80,
        )
        risk = calculate_risk_score(fused, 0.92, 0.02, 0.01)

        self.assertLess(fused, 0.35)
        self.assertGreater(risk, fused)


class SemanticScoringTests(unittest.TestCase):
    def setUp(self):
        self.profile = {
            "semantic_floor": 0.80,
            "score_weights": {
                "doc_semantic": 0.20,
                "coverage": 0.30,
                "confidence": 0.50,
            },
            "score_gate": {
                "base": 0.08,
                "coverage": 0.55,
                "confidence": 0.30,
                "evidence": 0.25,
                "low_cov_th": 0.05,
                "low_cov_cap": 0.08,
                "mid_cov_th": 0.08,
                "mid_conf_th": 0.35,
                "mid_cov_cap": 0.28,
                "low_evidence_cov_th": 0.10,
                "low_evidence_conf_th": 0.60,
                "low_evidence_cap": 0.10,
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
        }

    def test_semantic_risk_score_caps_low_evidence_same_topic_cases(self):
        semantic_excess = calculate_semantic_excess(0.95, self.profile["semantic_floor"])
        score = calculate_semantic_risk_score(
            self.profile,
            semantic_excess=semantic_excess,
            weighted_coverage=0.02,
            confidence=0.50,
        )

        self.assertLessEqual(score.risk_score, 0.08)
        self.assertGreater(score.base_score, 0.0)
        self.assertGreater(score.gate, 0.0)

    def test_semantic_pair_score_keeps_sparse_matches_bounded(self):
        score = calculate_semantic_pair_score(
            self.profile,
            effective_coverage=0.01,
            confidence=0.65,
            doc_semantic=0.96,
            paragraph_semantic=0.94,
            longest_run_ratio=0.004,
            top3_run_ratio=0.008,
        )

        self.assertGreater(score.semantic_signal, score.evidence_score)
        self.assertLessEqual(score.final_score, score.low_evidence_cap)
        self.assertLess(score.final_score, 0.35)


class CoarseAndGlobalScoringTests(unittest.TestCase):
    def test_coarse_score_blends_semantic_paragraph_and_lexical_signals(self):
        hotspot = calculate_paragraph_hotspot([0.30, 0.80, 0.90], top_k=2)
        score = compose_coarse_score(
            doc_semantic=0.76,
            paragraph_hotspot=hotspot,
            lexical_anchor=0.20,
        )

        self.assertGreater(hotspot, 0.70)
        self.assertGreater(score, 0.60)

    def test_global_score_caps_sparse_low_coverage_evidence(self):
        breakdown = calculate_global_score(
            raw_coverage=0.04,
            weighted_coverage=0.02,
            effective_coverage=0.02,
            confidence=0.40,
            continuity_top3=0.01,
            source_support=0.50,
            source_diversity=0.60,
        )

        self.assertLessEqual(breakdown.global_score, breakdown.low_evidence_cap)
        self.assertEqual(breakdown.score_level, "low")


class WindowScoringTests(unittest.TestCase):
    def test_window_topk_and_outlier_metrics_are_shared_scoring_helpers(self):
        scores = np.asarray([0.10, 0.85, 0.30, 0.70])
        top_indices = select_topk_indices(scores, 2)

        self.assertEqual(top_indices.tolist(), [1, 3])

        metrics = resolve_outlier_metrics(
            scores,
            peak_sim=0.85,
            profile_cfg={
                "outlier_std_k": 1.0,
                "outlier_percentile": 0.75,
                "outlier_percentile_min_windows": 2,
            },
        )

        self.assertTrue(metrics["used_percentile"])
        self.assertGreater(metrics["effective_threshold"], 0.0)


if __name__ == "__main__":
    unittest.main()
