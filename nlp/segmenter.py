"""中文分词器。

实现思路（不依赖第三方分词库，仅借用其词典数据）：

1. **基于词典的动态规划**：对句子构建 DAG，每个位置记录所有词典命中区间，
   用词频对数概率做最大概率路径（DP），得到全局最优的词典切分。
2. **HMM 未登录词识别**：对词典未覆盖的连续单字串，用字符级 BMES 模型
   （Viterbi 解码）重新切分，识别词典外的新词 / 人名等。
3. **混合切分**：英文、数字、标点按正则单独切分，中文串交给上述两步。

词典来自 :mod:`nlp.lexicon`（内置高频词 + jieba 词典数据源）。
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional

from .hmm import HMM
from .lexicon import load_dictionary


_CHINESE_RE = re.compile(r"[一-鿿]")
# 中文连续串
_CJK_BLOCK_RE = re.compile(r"([一-鿿]+)")
# 英文单词 / 数字 / 其它可见符号
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:[.\-@][a-zA-Z0-9]+)*")


class Segmenter:
    """基于词典 DP + HMM 的中文分词器。"""

    def __init__(self, dictionary: Optional[dict] = None):
        self.dictionary = dictionary if dictionary is not None else load_dictionary()
        self.max_word_len = max((len(w) for w in self.dictionary), default=4)
        self.total_freq = max(sum(self.dictionary.values()), 1)
        self._log_total = math.log(self.total_freq)
        # 未登录词默认频率（远低于常见词）
        self._oov_log = math.log(0.4 / self.total_freq)
        self._hmm = self._train_hmm()

    # -- HMM 训练 ---------------------------------------------------------
    def _train_hmm(self) -> HMM:
        """从词典生成 BMES 标注序列训练字符级 HMM。"""
        sequences = []
        for word in self.dictionary:
            if len(word) == 1:
                sequences.append([(word, "S")])
            else:
                tags = ["B"] + ["M"] * (len(word) - 2) + ["E"]
                sequences.append(list(zip(word, tags)))
        hmm = HMM(["B", "M", "E", "S"], add_k=0.001)
        hmm.train(sequences)
        return hmm

    # -- 对外接口 ---------------------------------------------------------
    def cut(self, text: str) -> list[str]:
        """把文本切分成词序列。"""
        if not text:
            return []
        tokens: list[str] = []
        for block in _CJK_BLOCK_RE.split(text):
            if not block:
                continue
            if _CHINESE_RE.search(block):
                tokens.extend(self._cut_cjk(block))
            else:
                tokens.extend(self._cut_ascii(block))
        return [t for t in tokens if t and not t.isspace()]

    def cut_sentences(self, sentences: Iterable[str]) -> list[list[str]]:
        return [self.cut(s) for s in sentences]

    # -- 非中文切分 -------------------------------------------------------
    @staticmethod
    def _cut_ascii(block: str) -> list[str]:
        tokens: list[str] = []
        pos = 0
        for match in _TOKEN_RE.finditer(block):
            if match.start() > pos:
                # 中间是标点 / 空白，逐字符切
                for ch in block[pos:match.start()]:
                    if not ch.isspace():
                        tokens.append(ch)
            tokens.append(match.group())
            pos = match.end()
        for ch in block[pos:]:
            if not ch.isspace():
                tokens.append(ch)
        return tokens

    # -- 中文切分：词典 DP + HMM ----------------------------------------
    def _cut_cjk(self, sentence: str) -> list[str]:
        dag = self._build_dag(sentence)
        route = self._best_route(sentence, dag)

        # 沿最优路径切分
        words: list[str] = []
        i = 0
        n = len(sentence)
        while i < n:
            end = route[i][0]
            word = sentence[i:end]
            words.append(word)
            i = end

        # 对未登录词（不在词典中的单字串）用 HMM 重新切分
        return self._refine_oov(words)

    def _build_dag(self, sentence: str) -> dict:
        n = len(sentence)
        dag: dict[int, list[int]] = {}
        for i in range(n):
            ends: list[int] = []
            for j in range(i + 1, min(i + self.max_word_len, n) + 1):
                if sentence[i:j] in self.dictionary:
                    ends.append(j)
            if not ends:
                ends.append(i + 1)
            dag[i] = ends
        return dag

    def _best_route(self, sentence: str, dag: dict) -> dict:
        """从右向左 DP，返回 route[i] = (最优 end, 累计对数概率)。"""
        n = len(sentence)
        route: dict[int, tuple[int, float]] = {}
        route[n] = (n, 0.0)
        for i in range(n - 1, -1, -1):
            best_end = dag[i][0]
            best_score = float("-inf")
            for end in dag[i]:
                word = sentence[i:end]
                if len(word) == 1 and word not in self.dictionary:
                    logp = self._oov_log
                else:
                    logp = math.log(self.dictionary.get(word, 1) / self.total_freq)
                score = logp + route[end][1]
                if score > best_score:
                    best_score = score
                    best_end = end
            route[i] = (best_end, best_score)
        return route

    def _refine_oov(self, words: list[str]) -> list[str]:
        """把连续的未登录单字合并，交给 HMM 切分。"""
        refined: list[str] = []
        oov_run: list[str] = []
        for word in words:
            if len(word) == 1 and word not in self.dictionary and _CHINESE_RE.match(word):
                oov_run.append(word)
            else:
                if oov_run:
                    refined.extend(self._hmm_segment("".join(oov_run)))
                    oov_run = []
                refined.append(word)
        if oov_run:
            refined.extend(self._hmm_segment("".join(oov_run)))
        return refined

    def _hmm_segment(self, chars: str) -> list[str]:
        """用 BMES HMM 对连续未知字符做词切分。"""
        tags = self._hmm.viterbi(list(chars))
        words: list[str] = []
        buf = ""
        for ch, tag in zip(chars, tags):
            if tag in ("B", "S"):
                if buf:
                    words.append(buf)
                buf = ch
            else:  # M / E
                buf += ch
        if buf:
            words.append(buf)
        return words
