import unittest

from engines.semantic.coarse_retrieval import (
    CoarseRetriever,
    CoarseRetrievalConfig,
    analyze_topic_concentration,
    compute_candidate_limit,
    select_candidate_indices,
)


class CoarseRetrievalTests(unittest.TestCase):
    def test_candidate_limit_expands_for_concentrated_topics(self):
        cfg = CoarseRetrievalConfig()

        self.assertEqual(
            compute_candidate_limit(80, cfg, topic_concentrated=False),
            12,
        )
        self.assertEqual(
            compute_candidate_limit(80, cfg, topic_concentrated=True),
            24,
        )

    def test_topic_concentration_detects_dense_same_topic_reference_pool(self):
        cfg = CoarseRetrievalConfig()
        scored_refs = [
            {"doc_semantic": 0.83},
            {"doc_semantic": 0.82},
            {"doc_semantic": 0.81},
            {"doc_semantic": 0.81},
            {"doc_semantic": 0.80},
            {"doc_semantic": 0.80},
            {"doc_semantic": 0.79},
        ]

        stats = analyze_topic_concentration(scored_refs, cfg)

        self.assertTrue(stats["is_concentrated"])
        self.assertGreater(stats["mean"], 0.80)
        self.assertLess(stats["std"], 0.03)

    def test_partial_config_override_is_normalized(self):
        cfg = CoarseRetrievalConfig.from_partial_dict(
            {
                "min_candidates": 0,
                "max_candidates": 3,
                "concentrated_min_candidates": 2,
                "concentrated_max_candidates": 1,
                "coarse_threshold": 1.4,
                "topic_std_threshold": -0.2,
                "paragraph_score_top_k": 99,
                "paragraph_max_count": 5,
            }
        )

        self.assertEqual(cfg.min_candidates, 1)
        self.assertEqual(cfg.max_candidates, 3)
        self.assertEqual(cfg.concentrated_min_candidates, 2)
        self.assertEqual(cfg.concentrated_max_candidates, 2)
        self.assertEqual(cfg.coarse_threshold, 1.0)
        self.assertEqual(cfg.topic_std_threshold, 0.0)
        self.assertEqual(cfg.paragraph_max_count, 5)
        self.assertEqual(cfg.paragraph_score_top_k, 5)

    def test_candidate_selection_uses_topk_threshold_lexical_and_paragraph_hotspots(self):
        cfg = CoarseRetrievalConfig(
            min_candidates=1,
            max_candidates=1,
            base_candidate_ratio=0.10,
            coarse_threshold=0.58,
            lexical_threshold=0.24,
            paragraph_hotspot_threshold=0.80,
            per_paragraph_top_m=1,
        )
        scored_refs = [
            {
                "coarse_score": 0.70,
                "lexical_anchor": 0.10,
                "best_target_paragraph_scores": [0.20, 0.10],
                "doc_semantic": 0.70,
            },
            {
                "coarse_score": 0.52,
                "lexical_anchor": 0.30,
                "best_target_paragraph_scores": [0.10, 0.20],
                "doc_semantic": 0.52,
            },
            {
                "coarse_score": 0.49,
                "lexical_anchor": 0.05,
                "best_target_paragraph_scores": [0.84, 0.15],
                "doc_semantic": 0.49,
            },
            {
                "coarse_score": 0.45,
                "lexical_anchor": 0.05,
                "best_target_paragraph_scores": [0.10, 0.82],
                "doc_semantic": 0.45,
            },
            {
                "coarse_score": 0.60,
                "lexical_anchor": 0.20,
                "best_target_paragraph_scores": [0.05, 0.05],
                "doc_semantic": 0.60,
            },
        ]

        selection = select_candidate_indices(
            scored_refs,
            target_paragraph_count=2,
            config=cfg,
            theme_stats={"mean": 0.60, "std": 0.10, "is_concentrated": False},
        )

        self.assertEqual(selection["candidate_limit"], 1)
        self.assertEqual(selection["candidate_indices"], [0, 4, 1, 2, 3])
        self.assertEqual(selection["candidate_ranks"][0], 1)
        self.assertIn("topk", selection["candidate_reasons"][0])
        self.assertIn("coarse_threshold", selection["candidate_reasons"][4])
        self.assertIn("lexical_anchor", selection["candidate_reasons"][1])
        self.assertIn("paragraph_hotspot_1", selection["candidate_reasons"][2])
        self.assertIn("paragraph_hotspot_2", selection["candidate_reasons"][3])

    def test_coarse_only_result_keeps_stage_metadata(self):
        result = CoarseRetriever.build_coarse_only_result(
            {
                "file": "sample.pdf",
                "coarse_score": 0.61,
                "doc_semantic": 0.77,
                "paragraph_hotspot": 0.58,
                "lexical_anchor": 0.22,
                "is_candidate": False,
                "candidate_rank": None,
                "coarse_rank": 3,
                "candidate_reason": "",
                "candidate_pool_size": 12,
                "reference_count": 80,
                "theme_mean": 0.81,
                "theme_std": 0.02,
                "topic_concentrated": True,
            },
            "balanced",
        )

        self.assertEqual(result["retrieval_stage"], "coarse_only")
        self.assertFalse(result["sim_bert_verified"])
        self.assertEqual(result["sim_bert"], 0.61)
        self.assertEqual(result["sim_bert_coarse_rank"], 3)
        self.assertTrue(result["retrieval_topic_concentrated"])


if __name__ == "__main__":
    unittest.main()
