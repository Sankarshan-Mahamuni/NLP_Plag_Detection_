#!/usr/bin/env python3
"""
NLP Plagiarism Detection CLI Tool
Usage:
    python detector.py <file1> <file2>
    python detector.py --demo
    python detector.py --batch <directory>
"""

import sys
import os
import argparse
from pathlib import Path

# Force UTF-8 stdout on Windows console
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from nlp_detector.similarity import analyze_similarity


def print_banner():
    banner = """
================================================================
          NLP Plagiarism & Paraphrase Detector
  Multi-Metric Analysis: TF-IDF + N-Gram Jaccard + LCS DP
================================================================
"""
    print(banner)


def display_report(name1: str, name2: str, report):
    print("=" * 64)
    print(f" Comparing: {name1}")
    print(f" Against  : {name2}")
    print("-" * 64)
    print(f" [*] TF-IDF Cosine Similarity : {report.tfidf_score:6.2f}%")
    print(f" [*] Bi-gram Jaccard Index    : {report.jaccard_score:6.2f}%")
    print(f" [*] Longest Common Seq (LCS) : {report.lcs_score:6.2f}%")
    print("-" * 64)
    print(f" [>] COMPOSITE PLAGIARISM     : {report.composite_score:6.2f}%")
    print(f" [!] VERDICT                  : {report.verdict}")
    print("=" * 64 + "\n")


def run_demo():
    print("\n[+] Running Interactive Test Matrix...\n")
    sample_dir = Path(__file__).parent / "sample_docs"
    
    doc_orig = sample_dir / "original.txt"
    doc_plag = sample_dir / "plagiarized.txt"
    doc_para = sample_dir / "paraphrased.txt"
    doc_unrel = sample_dir / "unrelated.txt"
    
    if not (doc_orig.exists() and doc_plag.exists()):
        print("Sample files not found. Creating samples first.")
        return
        
    orig_text = doc_orig.read_text(encoding="utf-8")
    plag_text = doc_plag.read_text(encoding="utf-8")
    para_text = doc_para.read_text(encoding="utf-8")
    unrel_text = doc_unrel.read_text(encoding="utf-8")
    
    print("TEST CASE 1: Original vs. Direct Plagiarism")
    display_report("original.txt", "plagiarized.txt", analyze_similarity(orig_text, plag_text))
    
    print("TEST CASE 2: Original vs. Paraphrased")
    display_report("original.txt", "paraphrased.txt", analyze_similarity(orig_text, para_text))
    
    print("TEST CASE 3: Original vs. Unrelated Domain")
    display_report("original.txt", "unrelated.txt", analyze_similarity(orig_text, unrel_text))


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="NLP Plagiarism & Similarity Detection Engine")
    parser.add_argument("files", nargs="*", help="Paths to two text files to compare")
    parser.add_argument("--demo", action="store_true", help="Run demonstrative sample comparison suite")
    parser.add_argument("--batch", type=str, help="Compare all text files in a directory pairwise")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
        return
        
    if args.batch:
        folder = Path(args.batch)
        if not folder.is_dir():
            print(f"Error: {args.batch} is not a valid directory.")
            sys.exit(1)
        txt_files = list(folder.glob("*.txt"))
        if len(txt_files) < 2:
            print("Need at least two .txt files in the directory to compare.")
            return
            
        print(f"Scanning {len(txt_files)} documents pairwise...\n")
        for i in range(len(txt_files)):
            for j in range(i + 1, len(txt_files)):
                t1 = txt_files[i].read_text(encoding="utf-8", errors="ignore")
                t2 = txt_files[j].read_text(encoding="utf-8", errors="ignore")
                rep = analyze_similarity(t1, t2)
                display_report(txt_files[i].name, txt_files[j].name, rep)
        return
        
    if len(args.files) == 2:
        p1, p2 = Path(args.files[0]), Path(args.files[1])
        if not p1.exists() or not p2.exists():
            print("Error: One or both files could not be found.")
            sys.exit(1)
            
        t1 = p1.read_text(encoding="utf-8", errors="ignore")
        t2 = p2.read_text(encoding="utf-8", errors="ignore")
        rep = analyze_similarity(t1, t2)
        display_report(p1.name, p2.name, rep)
    else:
        print("Usage:")
        print("  python detector.py <file1.txt> <file2.txt>")
        print("  python detector.py --demo")
        print("  python detector.py --batch sample_docs/")


if __name__ == "__main__":
    main()
