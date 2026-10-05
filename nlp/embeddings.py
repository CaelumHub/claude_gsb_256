"""词向量学习与可视化。

纯 Python 实现（无 numpy），完整链路：

1. 从语料统计词共现矩阵（滑动窗口）；
2. 计算 PPMI（正点互信息）加权；
3. 用**幂迭代 + 收缩法**做截断 SVD，得到稠密词向量；
4. 用 PCA 把高维向量投影到 2D / 3D 供前端散点图可视化；
5. 提供余弦相似度最近邻查询与 k-means 聚类（用于着色）。

算法复杂度与说明：截断 SVD 的幂迭代每次 O(n^2)，词表规模 n 默认 300，
rank 默认 30，适合演示语料；大语料可调低词表规模或 rank。
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Iterable, Optional

from .lexicon import STOPWORDS
from .segmenter import Segmenter


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    s = math.sqrt(sum(x * x for x in v))
    return s if s > 0 else 1.0


def _normalize(v: list[float]) -> list[float]:
    n = _norm(v)
    return [x / n for x in v]


def _matvec(mat, vec):
    return [_dot(row, vec) for row in mat]


def _matvec_t(mat, vec):
    n = len(mat)
    m = len(mat[0]) if n else 0
    out = [0.0] * m
    for i in range(n):
        row = mat[i]
        vi = vec[i]
        for j in range(m):
            out[j] += row[j] * vi
    return out


def truncated_svd(mat, k: int, iters: int = 50) -> tuple[list, list, list]:
    """对稠密矩阵做截断 SVD（幂迭代 + 收缩）。

    返回 ``(U, S, V)``：U/V 为左右奇异向量列表，S 为奇异值列表。
    """
    n = len(mat)
    if n == 0:
        return [], [], []
    k = min(k, n)
    B = [row[:] for row in mat]
    U, S, V = [], [], []
    rng = random.Random(0)

    for _ in range(k):
        v = [rng.gauss(0, 1) for _ in range(n)]
        v = _normalize(v)
        for _ in range(iters):
            u = _normalize(_matvec(B, v))
            v = _normalize(_matvec_t(B, u))
        u = _normalize(_matvec(B, v))
        s = _dot(u, _matvec(B, v))
        if s < 1e-9:
            break
        U.append(u)
        S.append(s)
        V.append(v)
        # 收缩：B -= s * u v^T
        for i in range(n):
            row = B[i]
            ui = u[i]
            for j in range(n):
                row[j] -= s * ui * v[j]
    return U, S, V


def _eigh_power(cov: list[list[float]], k: int, iters: int = 100):
    """对称矩阵的前 k 个最大特征向量（幂迭代 + 收缩）。"""
    n = len(cov)
    if n == 0:
        return []
    B = [row[:] for row in cov]
    vectors = []
    rng = random.Random(0)
    for _ in range(min(k, n)):
        v = [rng.gauss(0, 1) for _ in range(n)]
        v = _normalize(v)
        for _ in range(iters):
            v = _normalize(_matvec(B, v))
        lam = _dot(v, _matvec(B, v))
        vectors.append(v)
        for i in range(n):
            row = B[i]
            vi = v[i]
            for j in range(n):
                row[j] -= lam * vi * v[j]
    return vectors


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


def _kmeans(points: dict[str, list[float]], n_clusters: int,
            iters: int = 50, seed: int = 0) -> dict[str, int]:
    """简单 k-means，返回 词 -> 簇编号。"""
    words = list(points.keys())
    if len(words) <= n_clusters:
        return {w: i for i, w in enumerate(words)}
    rng = random.Random(seed)
    centers = [points[w][:] for w in rng.sample(words, n_clusters)]
    labels: dict[str, int] = {}

    for _ in range(iters):
        # 分配
        changed = False
        for w in words:
            v = points[w]
            best = max(range(n_clusters), key=lambda c: cosine(v, centers[c]))
            if labels.get(w) != best:
                changed = True
                labels[w] = best
        # 更新中心
        sums = [[0.0] * len(centers[0]) for _ in range(n_clusters)]
        counts = [0] * n_clusters
        for w, c in labels.items():
            v = points[w]
            for j in range(len(v)):
                sums[c][j] += v[j]
            counts[c] += 1
        for c in range(n_clusters):
            if counts[c]:
                centers[c] = [x / counts[c] for x in sums[c]]
        if not changed:
            break
    return labels


class WordEmbeddings:
    """基于 PPMI + 截断 SVD 的词向量模型。"""

    def __init__(self, segmenter: Optional[Segmenter] = None):
        self.segmenter = segmenter or Segmenter()
        self.vectors: dict[str, list[float]] = {}
        self.vocab: list[str] = []
        self.dim = 0
        self._word2id: dict[str, int] = {}
        self._svd_singular: list[float] = []

    # -- 训练 -------------------------------------------------------------
    def train(self, texts: Iterable[str], vocab_size: int = 300,
              dim: int = 30, window: int = 5, min_count: int = 1) -> "WordEmbeddings":
        """训练词向量。"""
        docs = [self._filter(self.segmenter.cut(t)) for t in texts]
        counter: Counter = Counter()
        for doc in docs:
            counter.update(doc)

        # 词表：按词频取前 vocab_size，且满足 min_count
        vocab = [w for w, c in counter.most_common(vocab_size) if c >= min_count]
        if not vocab:
            return self
        self.vocab = vocab
        self._word2id = {w: i for i, w in enumerate(vocab)}
        n = len(vocab)

        # 共现矩阵（对称，滑动窗口内衰减加权）
        cooc = [[0.0] * n for _ in range(n)]
        for doc in docs:
            for i, w in enumerate(doc):
                wi = self._word2id.get(w)
                if wi is None:
                    continue
                for j in range(i + 1, min(i + window, len(doc))):
                    cj = self._word2id.get(doc[j])
                    if cj is None:
                        continue
                    decay = 1.0 / (j - i)
                    cooc[wi][cj] += decay
                    cooc[cj][wi] += decay

        # PPMI
        total = sum(sum(row) for row in cooc)
        if total <= 0:
            return self
        row_sum = [sum(row) for row in cooc]
        ppmi = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                c = cooc[i][j]
                if c <= 0:
                    continue
                pmi = math.log((c * total) / (row_sum[i] * row_sum[j] + 1e-9))
                ppmi[i][j] = max(pmi, 0.0)

        # 截断 SVD，词向量 = U * S
        U, S, _ = truncated_svd(ppmi, dim)
        self.dim = len(S)
        self._svd_singular = S
        self.vectors = {}
        for i, w in enumerate(vocab):
            vec = [U[k][i] * S[k] for k in range(len(S))]
            self.vectors[w] = vec
        return self

    @staticmethod
    def _filter(words: list[str]) -> list[str]:
        return [w for w in words
                if w not in STOPWORDS and len(w) >= 1 and any(
                    "一" <= c <= "鿿" for c in w)]

    # -- 查询 -------------------------------------------------------------
    def nearest(self, word: str, k: int = 10) -> list[dict]:
        """返回与目标词最相似的 k 个词。"""
        if word not in self.vectors:
            return []
        v = self.vectors[word]
        scored = []
        for other, ov in self.vectors.items():
            if other == word:
                continue
            scored.append((other, cosine(v, ov)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{"word": w, "similarity": round(s, 4)} for w, s in scored[:k]]

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.vectors.get(a), self.vectors.get(b)
        if va is None or vb is None:
            return 0.0
        return cosine(va, vb)

    def project_2d(self, words: Optional[list[str]] = None) -> dict:
        """用 PCA 把词向量投影到 2D，返回 ``{词: [x, y]}``。"""
        subset = words or list(self.vectors.keys())
        subset = [w for w in subset if w in self.vectors]
        if len(subset) < 2 or self.dim == 0:
            return {w: [0.0, 0.0] for w in subset}

        points = [self.vectors[w] for w in subset]
        d = len(points[0])
        mean = [sum(p[i] for p in points) / len(points) for i in range(d)]
        centered = [[p[i] - mean[i] for i in range(d)] for p in points]

        # d x d 协方差
        cov = [[0.0] * d for _ in range(d)]
        for p in centered:
            for i in range(d):
                for j in range(d):
                    cov[i][j] += p[i] * p[j]

        comps = _eigh_power(cov, 2)
        result = {}
        for w, p in zip(subset, centered):
            result[w] = [_dot(p, comp) for comp in comps]
        return result

    def cluster(self, n_clusters: int = 5) -> dict[str, int]:
        return _kmeans(self.vectors, n_clusters)

    def stats(self) -> dict:
        return {
            "vocab_size": len(self.vocab),
            "dim": self.dim,
            "singular_values": [round(s, 4) for s in self._svd_singular[:10]],
        }
