"""文本处理公共工具。

提供句子切分、TF-IDF 计算、TextRank/PageRank 等被摘要与关键词提取
复用的基础算法。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from .lexicon import STOPWORDS


_SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+|(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """按中英文标点把文本切成句子，保留非空句。"""
    parts = _SENT_SPLIT_RE.split(text)
    sentences = []
    for p in parts:
        p = p.strip()
        if p:
            sentences.append(p)
    return sentences


def filter_stopwords(words: Iterable[str]) -> list[str]:
    """去掉停用词、单字符与纯符号词。"""
    result = []
    for w in words:
        if w in STOPWORDS:
            continue
        if len(w) < 2 and not _has_cjk(w):
            continue
        if not w.strip():
            continue
        result.append(w)
    return result


def _has_cjk(text: str) -> bool:
    return any("一" <= c <= "鿿" for c in text)


def compute_tfidf(documents: list[list[str]]) -> list[dict[str, float]]:
    """计算一组文档的 TF-IDF 向量。

    返回与 ``documents`` 等长的列表，每个元素是 ``{词: tfidf 权重}``。
    """
    n = len(documents)
    df: Counter = Counter()
    tf_list = []
    for doc in documents:
        tf = Counter(doc)
        tf_list.append(tf)
        for word in tf:
            df[word] += 1

    vectors = []
    for tf in tf_list:
        vec = {}
        doc_len = sum(tf.values()) or 1
        for word, count in tf.items():
            idf = math.log((n + 1) / (df[word] + 1)) + 1.0
            vec[word] = (count / doc_len) * idf
        vectors.append(vec)
    return vectors


def build_word_graph(documents: list[list[str]],
                     window: int = 4) -> tuple[dict[str, float], dict]:
    """构建词共现图，返回 (顶点权重, 邻接表)。

    在滑动窗口内共现的两个词之间建立带权边，权重随距离衰减。
    """
    graph: dict[str, dict[str, float]] = {}
    weight: dict[str, float] = {}

    for doc in documents:
        for i, w in enumerate(doc):
            weight[w] = weight.get(w, 0.0) + 1.0
            graph.setdefault(w, {})
            for j in range(i + 1, min(i + window, len(doc))):
                nxt = doc[j]
                if nxt == w:
                    continue
                decay = 1.0 / (j - i)
                graph[w][nxt] = graph[w].get(nxt, 0.0) + decay
                graph.setdefault(nxt, {})
                graph[nxt][w] = graph[nxt].get(w, 0.0) + decay
    return weight, graph


def pagerank(graph: dict[str, dict[str, float]],
             damping: float = 0.85,
             max_iter: int = 100,
             tol: float = 1e-6) -> dict[str, float]:
    """标准 PageRank，返回节点 -> 分数。"""
    nodes = list(graph.keys())
    n = len(nodes)
    if n == 0:
        return {}
    score = {node: 1.0 / n for node in nodes}

    for _ in range(max_iter):
        new_score = {}
        sink_sum = 0.0
        for node in nodes:
            neighbors = graph[node]
            out_sum = sum(neighbors.values())
            if out_sum == 0:
                sink_sum += score[node]
        for node in nodes:
            rank = (1 - damping) / n
            rank += damping * sink_sum / n
            for other in graph:
                out = sum(graph[other].values())
                if out > 0 and node in graph[other]:
                    rank += damping * score[other] * (graph[other][node] / out)
            new_score[node] = rank
        diff = sum(abs(new_score[node] - score[node]) for node in nodes)
        score = new_score
        if diff < tol:
            break
    return score


def similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """两个稀疏向量的余弦相似度。"""
    if not a or not b:
        return 0.0
    dot = sum(a.get(k, 0.0) * v for k, v in b.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
