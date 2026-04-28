import unittest

from reports import (
    build_report_payload,
    build_semantic_coarse_result,
    build_semantic_result,
    build_semantic_verified_result,
    build_traditional_result,
    sort_report_items,
)


class ReportItemTests(unittest.TestCase):
    def semantic_breakdown(self):
        return {
            "final_score": 0.42,
            "risk_score": 0.33,
            "doc_semantic": 0.81,
            "doc_semantic_excess": 0.21,
            "coverage": 0.12,
            "coverage_raw": 0.12,
            "coverage_weighted": 0.10,
            "coverage_effective": 0.11,
            "confidence": 0.72,
            "base_score": 0.38,
            "gate": 0.87,
            "hit_count": 2,
            "semantic_signal": 0.64,
            "evidence_score": 0.30,
            "continuity_bonus": 0.02,
            "continuity_longest": 0.04,
            "continuity_top3": 0.08,
            "low_evidence_cap": 1.0,
        }

    def test_build_semantic_result_keeps_frontend_contract(self):
        result = build_semantic_result(
            "temp/ref_sample.pdf",
            "balanced",
            self.semantic_breakdown(),
            [{"target_part": "a", "ref_part": "b", "score": 0.9}],
        )

        self.assertEqual(result["file"], "sample.pdf")
        self.assertEqual(result["engine"], "semantic")
        self.assertEqual(result["sim_bert"], 0.42)
        self.assertEqual(result["sim_bert_risk"], 0.33)
        self.assertEqual(result["sim_bert_hits"], 2)
        self.assertTrue(result["sim_bert_verified"])
        self.assertEqual(result["retrieval_stage"], "fine_verified")
        self.assertEqual(len(result["plagiarized_parts"]), 1)

    def test_build_semantic_coarse_and_verified_results_keep_retrieval_metadata(self):
        coarse_item = {
            "file": "ref_a.pdf",
            "coarse_score": 0.61,
            "doc_semantic": 0.77,
            "paragraph_hotspot": 0.58,
            "lexical_anchor": 0.22,
            "is_candidate": True,
            "candidate_rank": 2,
            "coarse_rank": 4,
            "candidate_reason": "topk",
            "candidate_pool_size": 8,
            "reference_count": 30,
            "theme_mean": 0.74,
            "theme_std": 0.04,
            "topic_concentrated": False,
        }

        coarse = build_semantic_coarse_result(coarse_item, "balanced")
        verified = build_semantic_verified_result(
            coarse_item,
            "balanced",
            self.semantic_breakdown(),
            [],
        )

        self.assertEqual(coarse["retrieval_stage"], "coarse_only")
        self.assertFalse(coarse["sim_bert_verified"])
        self.assertEqual(coarse["sim_bert_coarse_rank"], 4)
        self.assertEqual(verified["retrieval_stage"], "fine_verified")
        self.assertTrue(verified["sim_bert_verified"])
        self.assertEqual(verified["sim_bert_candidate_rank"], 2)
        self.assertEqual(verified["retrieval_reason"], "topk")

    def test_build_traditional_result_normalizes_file_and_scores(self):
        result = build_traditional_result(
            {
                "file": "tmp/ref_doc.txt",
                "sim_lsa": 0.20,
                "sim_tfidf": 0.12,
                "sim_soft": 0.09,
                "sim_hybrid": 0.24,
                "risk_score": 0.31,
                "traditional_lsa_components": 5,
                "traditional_lsa_components_effective": 4,
                "traditional_semantic_enabled": True,
                "traditional_semantic_mode": "vector",
                "traditional_semantic_vocab_size": 120,
                "traditional_semantic_vector_hits": 36,
                "traditional_semantic_vector_coverage": 0.30,
                "traditional_semantic_synonym_count": 18,
                "traditional_semantic_embeddings_configured": True,
                "traditional_semantic_embeddings_found": True,
                "plagiarized_parts": [{"target_part": "x"}],
            }
        )

        self.assertEqual(result["file"], "doc.txt")
        self.assertEqual(result["engine"], "traditional")
        self.assertEqual(result["risk_score"], 0.31)
        self.assertEqual(result["traditional_lsa_components"], 5)
        self.assertEqual(result["traditional_lsa_components_effective"], 4)
        self.assertTrue(result["traditional_semantic_enabled"])
        self.assertEqual(result["traditional_semantic_mode"], "vector")
        self.assertEqual(result["traditional_semantic_vector_hits"], 36)
        self.assertAlmostEqual(result["traditional_semantic_vector_coverage"], 0.30)
        self.assertEqual(len(result["plagiarized_parts"]), 1)

    def test_sort_report_items_and_payload_are_stable(self):
        bert_items = sort_report_items(
            [{"file": "b", "sim_bert": 0.2}, {"file": "a", "sim_bert": 0.8}],
            mode="bert",
        )
        traditional_items = sort_report_items(
            [
                {"file": "low", "risk_score": 0.1},
                {"file": "high", "risk_score": 0.9},
            ],
            mode="traditional",
        )
        payload = build_report_payload(bert_items, {"global_score": 0.7})

        self.assertEqual(bert_items[0]["file"], "a")
        self.assertEqual(traditional_items[0]["file"], "high")
        self.assertEqual(payload["items"][0]["file"], "a")
        self.assertEqual(payload["summary"]["global_score"], 0.7)


if __name__ == "__main__":
    unittest.main()
