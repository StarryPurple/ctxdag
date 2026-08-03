"""Answer-quality metrics for correctness evaluation (LongBench-style)."""

from __future__ import annotations

import re


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization, tolerant of Latin and CJK text."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_words(text: str) -> list[str]:
    """Split normalized text into words (CJK chars count as one word each)."""
    norm = normalize_answer(text)
    if not norm:
        return []
    return re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", norm)


def f1_score(prediction: str, reference: str) -> float:
    pred = tokenize_words(prediction)
    ref = tokenize_words(reference)
    if not pred or not ref:
        return 0.0
    common = sum(min(pred.count(w), ref.count(w)) for w in set(pred))
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def rouge_l_f1(prediction: str, reference: str) -> float:
    """ROUGE-L F1 over word sequences (LCS-based)."""
    pred = tokenize_words(prediction)
    ref = tokenize_words(reference)
    if not pred or not ref:
        return 0.0
    n, m = len(pred), len(ref)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n][m]
    if lcs == 0:
        return 0.0
    precision = lcs / n
    recall = lcs / m
    return 2 * precision * recall / (precision + recall)


def accuracy(prediction: str, references: list[str]) -> float:
    return 1.0 if normalize_answer(prediction) in {normalize_answer(r) for r in references} else 0.0


# LongBench-style metric per dataset family.
METRIC_F1 = {"2wikimqa", "hotpotqa", "musique", "multifieldqa_en", "qasper", "narrativeqa", "triviaqa", "dureader", "multifieldqa_zh"}
METRIC_ROUGE = {"multi_news", "qmsum", "samsum", "gov_report", "vcsum"}
METRIC_ACC = {"trec", "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh"}


def dataset_base(dataset: str) -> str:
    return dataset.removesuffix("_e")


def score_prediction(dataset: str, prediction: str, answers: list[str]) -> float:
    base = dataset_base(dataset)
    if base in METRIC_F1:
        return max(f1_score(prediction, a) for a in answers)
    if base in METRIC_ROUGE:
        return max(rouge_l_f1(prediction, a) for a in answers)
    if base in METRIC_ACC:
        return accuracy(prediction, answers)
    # default: F1
    return max(f1_score(prediction, a) for a in answers)
