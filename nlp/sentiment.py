"""情感分析与分类。

两套互补的方法：
1. **情感词典打分**：基于褒贬词 + 程度副词加权 + 否定翻转，逐词累加得到
   连续分值，再映射为正 / 负 / 中性。
2. **朴素贝叶斯分类器**：在一个内置的小型标注语料上训练多项式朴素贝叶斯
   （拉普拉斯平滑），给出三分类及后验概率。

输出综合结果，兼顾「可解释的词典打分」和「统计分类的置信度」。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

from .lexicon import (DEGREE_ADVERBS, NEGATION_WORDS, NEGATIVE_WORDS,
                      POSITIVE_WORDS)
from .segmenter import Segmenter


# 内置标注语料（用于朴素贝叶斯）
_LABELED = [
    ("这个产品非常好用，我很喜欢", "positive"),
    ("服务态度很差，让人失望", "negative"),
    ("今天天气一般般", "neutral"),
    ("质量优秀，值得推荐", "positive"),
    ("价格太贵了，不太划算", "negative"),
    ("这本书内容很精彩", "positive"),
    ("系统运行稳定，速度快", "positive"),
    ("客服回复很慢，体验糟糕", "negative"),
    ("外观漂亮，做工精致", "positive"),
    ("味道难吃，不会再来", "negative"),
    ("整体感觉还不错", "positive"),
    ("界面混乱，操作麻烦", "negative"),
    ("性能强劲，效率很高", "positive"),
    ("性价比很低，不建议购买", "negative"),
    ("他今天心情很好", "positive"),
    ("这部电影非常感人", "positive"),
    ("这部电影很无聊", "negative"),
    ("问题解决了，一切顺利", "positive"),
    ("出现了严重故障", "negative"),
    ("这个方案基本可行", "positive"),
    ("这个方案风险很大", "negative"),
    ("数据结果准确可靠", "positive"),
    ("数据存在明显错误", "negative"),
    ("体验普普通通", "neutral"),
    ("没有什么特别的", "neutral"),
    ("表现中规中矩", "neutral"),
    ("还不错，可以接受", "positive"),
    ("实在太差劲了", "negative"),
    ("非常满意的一次购物", "positive"),
    ("令人愤怒的体验", "negative"),
    ("效果立竿见影", "positive"),
    ("效果微乎其微", "negative"),
    ("态度还算可以", "neutral"),
    ("令人惊喜的进步", "positive"),
    ("令人担忧的下降", "negative"),
    ("各方面表现均衡", "neutral"),
]


class NaiveBayes:
    """多项式朴素贝叶斯（对数域 + 拉普拉斯平滑）。"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes: list[str] = []
        self.class_log_prior: dict[str, float] = {}
        self.feature_log_prob: dict[str, dict[str, float]] = {}
        self._vocab: set = set()

    def fit(self, documents: list[list[str]], labels: list[str]) -> "NaiveBayes":
        self.classes = sorted(set(labels))
        n = len(documents)
        class_counts = Counter(labels)
        self.class_log_prior = {
            c: math.log(class_counts[c] / n) for c in self.classes
        }
        self._vocab = {w for doc in documents for w in doc}

        # 每类中每个词的频次
        class_word_count: dict[str, Counter] = {c: Counter() for c in self.classes}
        for doc, label in zip(documents, labels):
            class_word_count[label].update(doc)

        for c in self.classes:
            total = sum(class_word_count[c].values())
            denom = total + self.alpha * len(self._vocab)
            self.feature_log_prob[c] = {
                w: math.log((class_word_count[c].get(w, 0) + self.alpha) / denom)
                for w in self._vocab
            }
        return self

    def predict_proba(self, tokens: list[str]) -> dict[str, float]:
        scores = {}
        for c in self.classes:
            score = self.class_log_prior[c]
            probs = self.feature_log_prob[c]
            for w in tokens:
                if w in probs:
                    score += probs[w]
            scores[c] = score
        # 对数概率 -> 概率（log-sum-exp 归一）
        return _softmax_log(scores)

    def predict(self, tokens: list[str]) -> str:
        return max(self.predict_proba(tokens), key=lambda k: self.predict_proba(tokens)[k])


def _softmax_log(log_scores: dict[str, float]) -> dict[str, float]:
    m = max(log_scores.values())
    exps = {k: math.exp(v - m) for k, v in log_scores.items()}
    total = sum(exps.values()) or 1.0
    return {k: v / total for k, v in exps.items()}


class SentimentAnalyzer:
    def __init__(self, segmenter: Optional[Segmenter] = None):
        self.segmenter = segmenter or Segmenter()
        self._nb = NaiveBayes(alpha=1.0)
        self._nb.fit(
            [self.segmenter.cut(text) for text, _ in _LABELED],
            [label for _, label in _LABELED],
        )

    def analyze(self, text: str) -> dict:
        words = self.segmenter.cut(text)
        score, pos_words, neg_words = self._lexicon_score(words)
        normalized = math.tanh(score / max(math.sqrt(len(words) or 1), 1.0))

        polarity = self._to_polarity(normalized)

        # 朴素贝叶斯分类
        probs = self._nb.predict_proba(words)
        nb_label = max(probs, key=probs.get)

        return {
            "score": round(normalized, 4),
            "raw_score": round(score, 4),
            "polarity": polarity,
            "positive_words": pos_words,
            "negative_words": neg_words,
            "classifier": {
                "label": nb_label,
                "probabilities": {k: round(v, 4) for k, v in probs.items()},
            },
            "confidence": round(probs[nb_label], 4),
        }

    # -- 词典打分 ---------------------------------------------------------
    def _lexicon_score(self, words: list[str]):
        score = 0.0
        pos_words = []
        neg_words = []
        for i, w in enumerate(words):
            sign = self._negation_sign(words, i)
            degree = self._degree(words, i)
            if w in POSITIVE_WORDS:
                score += sign * degree
                pos_words.append(w)
            elif w in NEGATIVE_WORDS:
                score -= sign * degree
                neg_words.append(w)
        return score, pos_words, neg_words

    @staticmethod
    def _negation_sign(words, i):
        for j in range(max(0, i - 2), i):
            if words[j] in NEGATION_WORDS:
                return -1.0
        return 1.0

    @staticmethod
    def _degree(words, i):
        for j in range(max(0, i - 2), i):
            if words[j] in DEGREE_ADVERBS:
                return DEGREE_ADVERBS[words[j]]
        return 1.0

    @staticmethod
    def _to_polarity(score: float, threshold: float = 0.15) -> str:
        if score >= threshold:
            return "positive"
        if score <= -threshold:
            return "negative"
        return "neutral"


# 中文标签
POLARITY_NAMES = {"positive": "积极", "negative": "消极", "neutral": "中性"}
