"""Similarity computation metrics: TF-IDF Cosine, Jaccard Index, and LCS."""

import math
from typing import Dict, List, Tuple
from collections import Counter
from nlp_detector.preprocessor import clean_text, tokenize, extract_ngrams


def compute_tf(tokens: List[str]) -> Dict[str, float]:
    """Compute Term Frequency for a list of tokens."""
    counts = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {word: count / total for word, count in counts.items()}


def compute_idf(doc_token_lists: List[List[str]]) -> Dict[str, float]:
    """Compute Inverse Document Frequency across a corpus."""
    n_docs = len(doc_token_lists)
    idf = {}
    all_words = set(w for doc in doc_token_lists for w in doc)
    
    for word in all_words:
        containing = sum(1 for doc in doc_token_lists if word in doc)
        idf[word] = math.log((1 + n_docs) / (1 + containing)) + 1.0
    return idf


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vector representations."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[k] * vec2[k] for k in intersection)
    
    sum1 = sum(v ** 2 for v in vec1.values())
    sum2 = sum(v ** 2 for v in vec2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    
    if not denominator:
        return 0.0
    return float(numerator / denominator)


def jaccard_similarity(tokens1: List[str], tokens2: List[str], n_gram: int = 2) -> float:
    """Compute Jaccard similarity coefficient on n-gram token sets."""
    set1 = extract_ngrams(tokens1, n=n_gram)
    set2 = extract_ngrams(tokens2, n=n_gram)
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return float(intersection / union) if union > 0 else 0.0


def longest_common_subsequence_ratio(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute Longest Common Subsequence (LCS) ratio between two token streams."""
    m, n = len(tokens1), len(tokens2)
    if m == 0 or n == 0:
        return 0.0
    
    # Space-optimized DP table for LCS length
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
            
    lcs_len = dp[n]
    return float((2.0 * lcs_len) / (m + n))


class PlagiarismReport:
    """Encapsulates plagiarism scoring metrics and qualitative verdict."""

    def __init__(self, tfidf_sim: float, jaccard_sim: float, lcs_ratio: float):
        self.tfidf_score = round(tfidf_sim * 100, 2)
        self.jaccard_score = round(jaccard_sim * 100, 2)
        self.lcs_score = round(lcs_ratio * 100, 2)
        
        # Weighted composite plagiarism score:
        # 50% TF-IDF Cosine + 30% N-gram Jaccard + 20% Sequence LCS
        self.composite_score = round(
            0.50 * self.tfidf_score + 0.30 * self.jaccard_score + 0.20 * self.lcs_score, 2
        )
        
        if self.composite_score >= 75.0:
            self.verdict = "CRITICAL: Severe Plagiarism Detected"
        elif self.composite_score >= 50.0:
            self.verdict = "WARNING: Substantial Content Overlap / Paraphrasing"
        elif self.composite_score >= 25.0:
            self.verdict = "MODERATE: Low-to-Moderate Overlap"
        else:
            self.verdict = "SAFE: Original Content (Negligible Similarity)"


def analyze_similarity(text1: str, text2: str) -> PlagiarismReport:
    """Performs full end-to-end multi-metric similarity analysis between two texts."""
    toks1 = tokenize(text1)
    toks2 = tokenize(text2)
    
    # Compute TF-IDF
    corpus = [toks1, toks2]
    idf = compute_idf(corpus)
    
    tf1 = compute_tf(toks1)
    tf2 = compute_tf(toks2)
    
    vec1 = {w: tf1[w] * idf.get(w, 1.0) for w in tf1}
    vec2 = {w: tf2[w] * idf.get(w, 1.0) for w in tf2}
    
    cos_sim = cosine_similarity(vec1, vec2)
    jaccard_sim = jaccard_similarity(toks1, toks2, n_gram=2)
    lcs_sim = longest_common_subsequence_ratio(toks1, toks2)
    
    return PlagiarismReport(cos_sim, jaccard_sim, lcs_sim)
