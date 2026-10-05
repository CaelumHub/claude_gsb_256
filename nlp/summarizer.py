"""抽取式文本摘要。

基于 TextRank：把句子当作图节点，句子相似度作为边权，用 PageRank
迭代计算句子重要度，再结合位置权重（首句倾向）选出 top-N 句，
并按原文顺序输出。

相似度使用句子词集（去停用词）的 TF-IDF 向量余弦相似度，兼顾语义相关。
"""

from __future__ import annotations

import math
from typing import Optional

from .segmenter import Segmenter
from .text import (build_word_graph, compute_tfidf, filter_stopwords,
                   pagerank, similarity, split_sentences)


class Summarizer:
    def __init__(self, segmenter: Optional[Segmenter] = None):
        self.segmenter = segmenter or Segmenter()

    def summarize(self, text: str, ratio: float = 0.3,
                  max_sentences: Optional[int] = None,
                  position_weight: float = 1.0) -> dict:
        """生成摘要。

        :param ratio: 抽取句数占比（当未指定 max_sentences 时使用）
        :param max_sentences: 最多抽取句数
        :param position_weight: 位置权重系数（>1 更偏向首句）
        """
        sentences = split_sentences(text)
        if not sentences:
            return {"summary": "", "sentences": [], "scores": []}
        if len(sentences) <= 1:
            return {"summary": sentences[0], "sentences": sentences, "scores": [1.0]}

        docs = [filter_stopwords(self.segmenter.cut(s)) for s in sentences]
        vectors = compute_tfidf(docs)

        # 构建句子相似度图
        graph: dict[int, dict[int, float]] = {i: {} for i in range(len(sentences))}
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i == j:
                    continue
                sim = similarity(vectors[i], vectors[j])
                if sim > 0.01:
                    graph[i][j] = sim

        # PageRank
        node_scores = pagerank(graph)

        # 结合位置权重（首句通常更重要）
        scores = []
        for i in range(len(sentences)):
            base = node_scores.get(i, 0.0)
            pos_bonus = position_weight * math.exp(-i / max(len(sentences), 1) * 2)
            scores.append(base + pos_bonus * 0.05)

        # 选择句数
        if max_sentences is None:
            max_sentences = max(1, int(len(sentences) * ratio))
        max_sentences = min(max_sentences, len(sentences))

        # 选 top-N，按原顺序输出
        ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        chosen = sorted(ranked[:max_sentences])

        summary = "".join(sentences[i] + "。" for i in chosen if i < len(sentences))
        if not summary.endswith(("。", "！", "？", ".", "!", "?")):
            summary = summary.rstrip("。")

        return {
            "summary": summary,
            "sentences": [{"index": i, "text": sentences[i],
                           "score": round(scores[i], 4),
                           "selected": i in chosen}
                          for i in range(len(sentences))],
            "top_indices": chosen,
        }
