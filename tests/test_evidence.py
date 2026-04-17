import unittest

from evidence import (
    GlobalEvidenceAggregator,
    normalize_evidence_span,
    normalize_evidence_spans,
    summarize_evidence,
)
from engines.semantic.global_evidence import GlobalEvidenceAggregator as LegacyAggregator


class EvidenceAdapterTests(unittest.TestCase):
    def test_normalize_evidence_span_adds_common_fields_and_keeps_engine_specific_metadata(self):
        span = normalize_evidence_span(
            {
                "target_part": "target text",
                "reference_part": "reference text",
                "score": 1.4,
                "score_tfidf": 0.31,
                "custom": "kept",
            },
            engine="traditional",
            source="ref.txt",
            default_match_type="traditional_window",
        )

        self.assertEqual(span["score"], 1.0)
        self.assertEqual(span["confidence"], 1.0)
        self.assertEqual(span["engine"], "traditional")
        self.assertEqual(span["source"], "ref.txt")
        self.assertEqual(span["match_type"], "traditional_window")
        self.assertEqual(span["ref_part"], "reference text")
        self.assertEqual(span["score_tfidf"], 0.31)
        self.assertEqual(span["metadata"]["custom"], "kept")

    def test_normalize_evidence_spans_ignores_non_mapping_items(self):
        spans = normalize_evidence_spans(
            [
                {"target_part": "a", "ref_part": "b", "score": 0.6},
                "not-a-span",
            ],
            engine="semantic",
        )

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["engine"], "semantic")


class EvidenceMetricsTests(unittest.TestCase):
    def test_summarize_evidence_deduplicates_intervals_and_keeps_confidence(self):
        summary = summarize_evidence(
            [
                {
                    "target_start": 0,
                    "target_end": 40,
                    "score": 0.90,
                    "confidence": 0.90,
                    "length": 40,
                },
                {
                    "target_start": 20,
                    "target_end": 60,
                    "score": 0.80,
                    "confidence": 0.80,
                    "length": 40,
                },
            ],
            target_len=100,
        )

        data = summary.to_dict()
        self.assertAlmostEqual(data["coverage_raw"], 0.60, places=6)
        self.assertGreater(data["coverage_weighted"], 0.45)
        self.assertGreater(data["coverage_effective"], data["coverage_weighted"])
        self.assertEqual(data["interval_hit_count"], 2)
        self.assertEqual(data["merged_hit_count"], 1)
        self.assertGreater(data["confidence"], 0.75)


class GlobalEvidenceAggregatorTests(unittest.TestCase):
    def test_global_aggregator_accepts_semantic_and_traditional_evidence(self):
        aggregator = GlobalEvidenceAggregator(target_normalizer=lambda text: text or "")
        summary = aggregator.aggregate(
            "abcdefghij" * 20,
            [
                {
                    "file": "semantic.pdf",
                    "sim_bert": 0.72,
                    "sim_bert_risk": 0.75,
                    "sim_bert_hits": 1,
                    "plagiarized_parts": [
                        {
                            "target_start": 0,
                            "target_end": 50,
                            "score": 0.88,
                            "confidence": 0.86,
                            "length": 50,
                        }
                    ],
                },
                {
                    "file": "traditional.pdf",
                    "sim_hybrid": 0.42,
                    "risk_score": 0.50,
                    "plagiarized_parts": [
                        {
                            "target_part": "local traditional hit",
                            "ref_part": "reference hit",
                            "score": 0.44,
                            "score_tfidf": 0.39,
                            "length": 60,
                        }
                    ],
                },
            ],
            bert_profile="balanced",
            retrieval_strategy="mixed",
        )

        self.assertEqual(summary["retrieval_stage"], "global_summary")
        self.assertEqual(summary["global_verified_source_count"], 2)
        self.assertEqual(summary["global_interval_hit_count"], 1)
        self.assertEqual(summary["top_sources"][0]["engine"], "semantic")
        self.assertIn(
            "traditional",
            {item["engine"] for item in summary["top_sources"]},
        )

    def test_legacy_global_aggregator_import_still_resolves(self):
        self.assertIs(LegacyAggregator, GlobalEvidenceAggregator)


if __name__ == "__main__":
    unittest.main()
