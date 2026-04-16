"""Command-line entry point for the traditional plagiarism detector.

The implementation lives in ``engines.traditional.system``.
"""

import argparse
import glob
import os

from engines.traditional.system import PlagiarismDetectorSystem


def _resolve_reference_files(target_file: str, reference_input: str):
    if os.path.isdir(reference_input):
        reference_files = glob.glob(os.path.join(reference_input, "*.txt"))
        return [
            path
            for path in reference_files
            if os.path.abspath(path) != os.path.abspath(target_file)
        ]

    if os.path.exists(reference_input):
        return [reference_input]

    print(f"Reference path does not exist: {reference_input}")
    print("Trying default data directory...")
    reference_files = glob.glob(os.path.join("data", "*.txt"))
    return [
        path
        for path in reference_files
        if os.path.abspath(path) != os.path.abspath(target_file)
    ]


def main():
    parser = argparse.ArgumentParser(description="Document similarity and plagiarism detection CLI")
    parser.add_argument("target", nargs="?", help="Target document path")
    parser.add_argument("reference", nargs="?", help="Reference document path or directory")
    parser.add_argument("--stopwords", default="dicts/stopwords.txt", help="Stopwords file path")
    parser.add_argument("--lsa_dim", type=int, default=3, help="LSA dimension")
    parser.add_argument(
        "--semantic_embeddings",
        default="dicts/embeddings/fasttext_zh.vec",
        help="Word vector model path for traditional soft semantic scoring",
    )
    parser.add_argument("--semantic_threshold", type=float, default=0.55)
    parser.add_argument("--semantic_weight", type=float, default=0.35)
    args = parser.parse_args()

    target_file = args.target if args.target else "data/AI医疗_原文.txt"
    reference_input = args.reference if args.reference else "data/"
    reference_files = _resolve_reference_files(target_file, reference_input)

    if not os.path.exists(target_file):
        print(f"Target document does not exist: {target_file}")
        raise SystemExit(1)

    if not reference_files:
        print("No reference documents found.")
        raise SystemExit(1)

    detector = PlagiarismDetectorSystem(
        stopwords_path=args.stopwords,
        lsa_components=args.lsa_dim,
        synonyms_path="dicts/synonyms.txt",
        semantic_embeddings_path=args.semantic_embeddings,
        semantic_threshold=args.semantic_threshold,
        semantic_weight=args.semantic_weight,
    )
    results = detector.check_similarity(target_file, reference_files)
    detector.print_report(target_file, results)


if __name__ == "__main__":
    main()


__all__ = ["PlagiarismDetectorSystem", "main"]
