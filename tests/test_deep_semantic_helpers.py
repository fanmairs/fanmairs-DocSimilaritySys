import unittest

from deep_semantic import DeepSemanticEngine


class DeepSemanticHelperTests(unittest.TestCase):
    def setUp(self):
        self.engine = DeepSemanticEngine.__new__(DeepSemanticEngine)
        self.strict_profile = {
            "final_score": {
                "semantic_weight": 0.24,
                "evidence_weight": 0.76,
                "semantic_center": 0.84,
                "semantic_scale": 0.07,
                "coverage_gain": 9.0,
                "low_evidence_cap_base": 0.10,
                "low_evidence_cap_gain": 0.16,
                "continuity_boost": 0.05,
            }
        }

    def test_effective_coverage_uses_weighted_coverage_as_baseline(self):
        effective = self.engine._calculate_effective_coverage(
            raw_coverage=0.60,
            weighted_coverage=0.20,
            confidence=0.00,
        )

        self.assertAlmostEqual(effective, 0.28, places=6)
        self.assertGreaterEqual(effective, 0.20)
        self.assertLessEqual(effective, 0.60)

    def test_technical_spans_are_downgraded_instead_of_rejected(self):
        self.engine._extract_entities = lambda text: set()
        self.engine._extract_tags = lambda text: set()
        self.engine._get_skeleton = lambda text: text
        self.engine._is_formula_explanation = lambda text: False

        item1 = {
            "text": "Alpha beta gamma. 2024. 2025. 2026. 2027. 2028. module version 123456.",
            "start": 0,
            "end": 74,
        }
        item2 = {
            "text": "Alpha beta gamma. 2024. 2025. 2026. 2027. 2028. system version 123450.",
            "start": 10,
            "end": 84,
        }

        scored = self.engine._score_window_candidate(
            item1,
            item2,
            raw_sim=0.91,
            outlier_threshold=0.86,
            profile_cfg={"min_window_chars": 5},
        )

        self.assertIsNotNone(scored)
        self.assertLess(scored["score"], 0.91)
        self.assertIn("target_many_periods", scored["rule_flags"])
        self.assertIn("ref_many_periods", scored["rule_flags"])

    def test_numeric_table_spans_are_rejected_as_evidence(self):
        item1 = {
            "text": "3.12 0.162 0.071 0.148 2.282 0.025* 3.88 0.428 0.152 0.386 2.816",
            "start": 0,
            "end": 84,
        }
        item2 = {
            "text": "12.314 12.758 12.815 13.981 14.020 2.138 2.228 4.471 4.472 8.323",
            "start": 10,
            "end": 88,
        }

        scored = self.engine._score_window_candidate(
            item1,
            item2,
            raw_sim=0.92,
            outlier_threshold=0.86,
            profile_cfg={"min_window_chars": 5},
        )

        self.assertIsNone(scored)

    def test_realistic_score_no_longer_crushes_high_semantic_sparse_matches(self):
        final_score, effective_coverage, semantic_signal, evidence_score, continuity_bonus, low_evidence_cap = self.engine._calculate_realistic_score(
            profile_name="strict",
            profile_cfg=self.strict_profile,
            raw_coverage=0.008418708240534522,
            weighted_coverage=0.006020115367483296,
            confidence=0.6639396920967763,
            doc_semantic=0.96142578125,
            paragraph_semantic=0.955,
            hit_count=2,
            longest_run_ratio=0.0042,
            top3_run_ratio=0.0084,
        )

        self.assertGreater(final_score, 0.15)
        self.assertLess(final_score, 0.30)
        self.assertGreater(semantic_signal, evidence_score)
        self.assertGreater(low_evidence_cap, 0.20)
        self.assertGreater(effective_coverage, 0.006)
        self.assertGreaterEqual(continuity_bonus, 0.0)

    def test_continuity_features_capture_longest_and_top3_runs(self):
        continuity = self.engine._calculate_continuity_features(
            plagiarized_parts=[
                {"target_start": 0, "target_end": 20},
                {"target_start": 30, "target_end": 50},
                {"target_start": 80, "target_end": 110},
                {"target_start": 45, "target_end": 60},
            ],
            target_len=200,
        )

        self.assertAlmostEqual(continuity["longest_run_ratio"], 30 / 200, places=6)
        self.assertAlmostEqual(continuity["top3_run_ratio"], 80 / 200, places=6)
        self.assertEqual(continuity["merged_hit_count"], 3)


if __name__ == "__main__":
    unittest.main()
