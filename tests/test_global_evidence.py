import unittest

from deep_semantic import DeepSemanticEngine
from global_evidence import GlobalEvidenceAggregator


class GlobalEvidenceAggregatorTests(unittest.TestCase):
    def setUp(self):
        self.engine = DeepSemanticEngine.__new__(DeepSemanticEngine)
        self.aggregator = GlobalEvidenceAggregator(self.engine)

    def test_aggregate_deduplicates_overlapping_source_intervals(self):
        summary = self.aggregator.aggregate(
            "abcdefghij" * 20,
            [
                {
                    "file": "ref_a.txt",
                    "sim_bert": 0.72,
                    "sim_bert_risk": 0.74,
                    "sim_bert_hits": 2,
                    "plagiarized_parts": [
                        {
                            "target_start": 0,
                            "target_end": 40,
                            "confidence": 0.90,
                            "score": 0.88,
                            "length": 40,
                        },
                        {
                            "target_start": 42,
                            "target_end": 70,
                            "confidence": 0.82,
                            "score": 0.80,
                            "length": 28,
                        },
                    ],
                },
                {
                    "file": "ref_b.txt",
                    "sim_bert": 0.68,
                    "sim_bert_risk": 0.70,
                    "sim_bert_hits": 1,
                    "plagiarized_parts": [
                        {
                            "target_start": 20,
                            "target_end": 60,
                            "confidence": 0.86,
                            "score": 0.84,
                            "length": 40,
                        }
                    ],
                },
            ],
            bert_profile="balanced",
            reference_count=8,
            candidate_count=3,
        )

        self.assertEqual(summary["retrieval_stage"], "global_summary")
        self.assertEqual(summary["global_reference_count"], 8)
        self.assertEqual(summary["global_candidate_count"], 3)
        self.assertEqual(summary["global_verified_source_count"], 2)
        self.assertEqual(summary["global_hit_count"], 3)
        self.assertGreater(summary["global_coverage_raw"], 0.30)
        self.assertLess(summary["global_coverage_raw"], 0.36)
        self.assertGreater(summary["global_source_diversity"], 0.40)
        self.assertGreater(summary["global_score"], 0.25)

    def test_aggregate_caps_sparse_low_coverage_evidence(self):
        summary = self.aggregator.aggregate(
            "abcdefghij" * 30,
            [
                {
                    "file": "ref_sparse.txt",
                    "sim_bert": 0.61,
                    "sim_bert_risk": 0.64,
                    "sim_bert_hits": 1,
                    "plagiarized_parts": [
                        {
                            "target_start": 5,
                            "target_end": 11,
                            "confidence": 0.53,
                            "score": 0.55,
                            "length": 6,
                        }
                    ],
                }
            ],
            bert_profile="strict",
        )

        self.assertLess(summary["global_coverage_effective"], 0.05)
        self.assertLessEqual(summary["global_score"], summary["global_score_raw"])
        self.assertLessEqual(summary["global_score"], summary["global_low_evidence_cap"])
        self.assertEqual(summary["global_score_level"], "low")

    def test_aggregate_handles_sources_without_coordinate_hits(self):
        summary = self.aggregator.aggregate(
            "abcdefghij" * 15,
            [
                {
                    "file": "ref_structural.txt",
                    "sim_bert": 0.44,
                    "sim_bert_risk": 0.46,
                    "sim_bert_hits": 1,
                    "plagiarized_parts": [
                        {
                            "target_start": None,
                            "target_end": None,
                            "confidence": 0.72,
                            "score": 0.74,
                            "length": 60,
                        }
                    ],
                }
            ],
            bert_profile="balanced",
        )

        self.assertEqual(summary["global_interval_hit_count"], 0)
        self.assertEqual(summary["global_dedup_span_count"], 0)
        self.assertGreater(summary["global_confidence"], 0.0)
        self.assertLess(summary["global_score"], 0.35)


if __name__ == "__main__":
    unittest.main()
