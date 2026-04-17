import unittest

from engines.base import EnginePairResult, UnsupportedEngineError
from engines.factory import create_engine, resolve_engine_kind
from engines.semantic import CoarseRetrievalConfig, DeepSemanticEngine
from engines.traditional import WhiteBoxTFIDF
from engines.semantic.engine import SemanticEngine


class FakeTraditionalSystem:
    def __init__(self):
        self.calls = []

    def read_document(self, filepath, preview_mode=False):
        return f"read:{filepath}:{preview_mode}"

    def clean_academic_noise(self, text):
        return text.replace("[1]", "")

    def check_similarity(self, target_file, reference_files, body_mode=False):
        self.calls.append((target_file, reference_files, body_mode))
        return [{"file": reference_files[0], "sim_hybrid": 0.42}]


class FakeSemanticBackend:
    def _normalize_text(self, text):
        return " ".join((text or "").split())

    def _build_windows(self, text):
        return [{"text": part} for part in text.split(" ") if part]

    def sliding_window_check(self, target_text, reference_text, threshold_profile="balanced"):
        return [{"target_part": target_text[:3], "reference_part": reference_text[:3], "score": 0.88}]

    def score_document_pair(self, target_text, reference_text, plagiarized_parts, threshold_profile="balanced"):
        return {
            "final_score": 0.31,
            "risk_score": 0.27,
            "coverage": 0.12,
            "confidence": 0.66,
            "hit_count": len(plagiarized_parts),
        }


class EngineFactoryTests(unittest.TestCase):
    def test_resolve_engine_kind_accepts_aliases(self):
        self.assertEqual(resolve_engine_kind("tfidf"), "traditional")
        self.assertEqual(resolve_engine_kind("BGE"), "semantic")
        self.assertEqual(resolve_engine_kind("deep-semantic"), "semantic")

    def test_resolve_engine_kind_rejects_unknown_alias(self):
        with self.assertRaises(UnsupportedEngineError):
            resolve_engine_kind("unknown")

    def test_create_traditional_engine_with_injected_system(self):
        fake_system = FakeTraditionalSystem()
        engine = create_engine("traditional", system=fake_system)

        result = engine.compare_files("target.txt", ["ref.txt"], body_mode=True)

        self.assertEqual(result[0]["sim_hybrid"], 0.42)
        self.assertEqual(fake_system.calls, [("target.txt", ["ref.txt"], True)])
        self.assertEqual(engine.clean_academic_noise("hello[1]"), "hello")

    def test_semantic_engine_compare_pair_returns_common_result(self):
        engine = SemanticEngine(engine=FakeSemanticBackend())

        result = engine.compare_pair("目标文本", "参考文本", reference="ref.pdf")

        self.assertIsInstance(result, EnginePairResult)
        self.assertEqual(result.engine, "semantic")
        self.assertEqual(result.reference, "ref.pdf")
        self.assertAlmostEqual(result.score, 0.31)
        self.assertAlmostEqual(result.risk_score, 0.27)
        self.assertEqual(len(result.matches), 1)

    def test_semantic_engine_estimates_windows_without_model_load(self):
        engine = create_engine("bge", engine=FakeSemanticBackend())

        self.assertEqual(engine.estimate_window_count("一 二 三"), 3)
        self.assertEqual(engine.estimate_window_count(""), 0)

    def test_package_exports_core_engine_classes(self):
        cfg = CoarseRetrievalConfig(max_candidates=3)
        vectorizer = WhiteBoxTFIDF()

        self.assertEqual(cfg.max_candidates, 3)
        self.assertTrue(hasattr(DeepSemanticEngine, "sliding_window_check"))
        self.assertEqual(vectorizer.vocab, {})


if __name__ == "__main__":
    unittest.main()
