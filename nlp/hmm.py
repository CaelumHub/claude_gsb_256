"""通用隐马尔可夫模型 (HMM)。

为分词和词性标注提供统一的概率图模型基础设施：

- 监督训练：从带标注序列统计转移 / 发射概率，采用加一平滑（add-k）避免零概率。
- Viterbi 解码：给定观测序列，求最优状态序列。
- 后向 A* / Beam 可选（保留接口）。

所有概率都保存在对数域，避免连乘下溢。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Optional, Sequence


_NEG_INF = float("-inf")


class HMM:
    """离散 HMM。

    :param states: 状态集合（如 BMES 或词性标签集）
    :param add_k: 平滑参数，越大越平滑
    """

    def __init__(self, states: Sequence[str], add_k: float = 0.01):
        self.states = list(states)
        self.state_index = {s: i for i, s in enumerate(self.states)}
        self.add_k = add_k

        self.start = [0.0] * len(self.states)
        self.trans = [[0.0] * len(self.states) for _ in self.states]
        self.emit = defaultdict(lambda: [0.0] * len(self.states))
        self.trained = False

    # -- 训练 -------------------------------------------------------------
    def train(self, sequences: Iterable[Sequence[tuple]],
              start: bool = True, trans: bool = True, emit: bool = True) -> "HMM":
        """从 ``(观测, 状态)`` 序列对中统计计数并归一化。

        每条序列形如 ``[(obs_1, state_1), (obs_2, state_2), ...]``。
        """
        start_cnt = [self.add_k] * len(self.states)
        trans_cnt = [[self.add_k] * len(self.states) for _ in self.states]
        emit_cnt: dict = defaultdict(lambda: [self.add_k] * len(self.states))

        for seq in sequences:
            if not seq:
                continue
            first_state = seq[0][1]
            if first_state in self.state_index:
                start_cnt[self.state_index[first_state]] += 1
            for i, (obs, state) in enumerate(seq):
                if state not in self.state_index:
                    continue
                si = self.state_index[state]
                emit_cnt[obs][si] += 1
                if i + 1 < len(seq):
                    nxt = seq[i + 1][1]
                    if nxt in self.state_index:
                        trans_cnt[si][self.state_index[nxt]] += 1

        if start:
            self.start = _normalize(start_cnt)
        if trans:
            self.trans = [_normalize(row) for row in trans_cnt]
        if emit:
            self.emit = {o: _normalize(cnt) for o, cnt in emit_cnt.items()}
        self.trained = True
        return self

    # -- 概率查询（对数域） ----------------------------------------------
    def start_logp(self, state: str) -> float:
        idx = self.state_index.get(state)
        if idx is None:
            return _NEG_INF
        return _safe_log(self.start[idx])

    def trans_logp(self, prev: str, nxt: str) -> float:
        i, j = self.state_index.get(prev), self.state_index.get(nxt)
        if i is None or j is None:
            return _NEG_INF
        return _safe_log(self.trans[i][j])

    def emit_logp(self, obs, state: str) -> float:
        idx = self.state_index.get(state)
        if idx is None:
            return _NEG_INF
        probs = self.emit.get(obs)
        if probs is None:
            # 未见观测：回退到均匀发射
            return _safe_log(self.add_k / (sum(self.emit[obs] for obs in []) + 1))
        return _safe_log(probs[idx])

    # -- Viterbi 解码 -----------------------------------------------------
    def viterbi(self, observations: Sequence) -> list[str]:
        """返回与观测序列最匹配的状态序列。"""
        n = len(observations)
        k = len(self.states)
        if n == 0:
            return []

        # 使用对数概率
        viterbi = [[_NEG_INF] * k for _ in range(n)]
        backptr = [[-1] * k for _ in range(n)]

        for s, state in enumerate(self.states):
            viterbi[0][s] = self.start_logp(state) + self.emit_logp(observations[0], state)

        for t in range(1, n):
            obs = observations[t]
            for s, state in enumerate(self.states):
                best = _NEG_INF
                best_prev = -1
                for p in range(k):
                    score = (viterbi[t - 1][p]
                             + self.trans_logp(self.states[p], state))
                    if score > best:
                        best = score
                        best_prev = p
                viterbi[t][s] = best + self.emit_logp(obs, state)
                backptr[t][s] = best_prev

        # 回溯
        last = max(range(k), key=lambda s: viterbi[n - 1][s])
        path = [self.states[last]]
        for t in range(n - 1, 0, -1):
            last = backptr[t][last]
            path.append(self.states[last])
        path.reverse()
        return path


def _normalize(counts: Sequence[float]) -> list[float]:
    total = sum(counts)
    if total <= 0:
        return [1.0 / len(counts)] * len(counts)
    return [c / total for c in counts]


def _safe_log(p: float) -> float:
    if p <= 0:
        return _NEG_INF
    return math.log(p)
