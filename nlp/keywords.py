"""关键词提取。

TF-IDF 与 TextRank 的混合：
- TF-IDF 捕捉「词在当前文档中相对于背景的重要度」；
- TextRank 捕捉「词在共现网络中的中心性」；
两者归一化后加权融合，并保留名词 / 动词 / 形容词等实义词作为候选。
"""

from __future__ import annotations

from typing import Optional

from .pos import POSTagger
from .segmenter import Segmenter
from .text import build_word_graph, compute_tfidf, filter_stopwords, pagerank


# 允许作为关键词的词性
_KEY_POS = {"n", "v", "a", "ns", "nr", "nt", "vn"}


class KeywordExtractor:
    def __init__(self, tagger: Optional[POSTagger] = None):
        self.tagger = tagger or POSTagger()
        self.segmenter = self.tagger.segmenter

    def extract(self, text: str, top_k: int = 10,
                method: str = "hybrid") -> dict:
        """提取关键词。

        :param top_k: 返回数量
        :param method: ``hybrid``（默认）/ ``tfidf`` / ``textrank``
        """
        pairs = self.tagger.tag(text)
        words = [w for w, t in pairs if t in _KEY_POS and w not in _KEY_STOP]
        words = filter_stopwords(words)
        if not words:
            return {"keywords": []}

        # TF-IDF（把全文当作单文档，IDF 用词频近似）
        tfidf = compute_tfidf([words])[0]
        weight, graph = build_word_graph([words])
        rank = pagerank(graph)

        # 归一化
        tfidf_norm = _normalize(tfidf)
        rank_norm = _normalize(rank)

        scored = []
        for w in words:
            if method == "tfidf":
                score = tfidf_norm.get(w, 0.0)
            elif method == "textrank":
                score = rank_norm.get(w, 0.0)
            else:
                score = 0.5 * tfidf_norm.get(w, 0.0) + 0.5 * rank_norm.get(w, 0.0)
            scored.append((w, score, weight.get(w, 0)))

        # 去重并按分数排序
        seen = set()
        unique = []
        for w, s, f in sorted(scored, key=lambda x: x[1], reverse=True):
            if w in seen:
                continue
            seen.add(w)
            unique.append({"word": w, "score": round(s, 4), "freq": f})

        return {"keywords": unique[:top_k], "method": method}


# 额外需要排除的功能词（补充 stopwords）
_KEY_STOP = {"一个", "一种", "这个", "那个", "这些", "那些", "自己", "可以",
             "可能", "应该", "需要", "进行", "表示", "认为", "知道", "觉得",
             "开始", "结束", "完成", "成为", "使用", "提供", "支持", "包括"}


def _normalize(mapping: dict[str, float]) -> dict[str, float]:
    top = max(mapping.values()) if mapping else 1.0
    if top <= 0:
        return {}
    return {k: v / top for k, v in mapping.items()}
