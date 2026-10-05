"""句法分析：依存句法 + 短语结构句法。

- :class:`DependencyParser`：基于词性序列的规则式依存分析，确定每个词的
  支配词（head）和依存关系，输出一棵有向树。核心动词作为根节点。
- :class:`ConstituencyParser`：基于 CYK 算法的短语结构分析，用一个小型
  上下文无关文法（标签级 PCFG）自底向上解析出成分树。

两者都输入分词 + 词性标注结果，输出树结构供前端可视化。
"""

from __future__ import annotations

import math
from typing import Optional

from .pos import POSTagger
from .segmenter import Segmenter


# 依存关系中文标签
DEP_REL_NAMES = {
    "root": "核心", "nsubj": "主语", "dobj": "宾语", "amod": "定语修饰",
    "nummod": "数量修饰", "advmod": "状语修饰", "conj": "并列", "mark": "助词",
    "case": "介词", "clf": "量词", "lobj": "方位", "tmod": "时间修饰",
    "compound": "复合词", "punct": "标点", "attr": "系词补语", "dep": "其它",
}

# 短语结构非终结符中文标签
PHRASE_NAMES = {
    "S": "句子", "NP": "名词短语", "VP": "动词短语", "AP": "形容词短语",
    "PP": "介词短语", "ADVP": "副词短语", "P": "介词", "CONJ": "连词",
    "U": "助词", "TP": "时间短语",
}


class DependencyParser:
    """规则式依存句法分析器。"""

    def __init__(self, tagger: Optional[POSTagger] = None):
        self.tagger = tagger or POSTagger()

    def parse(self, text) -> dict:
        """输入句子字符串或 [(词, 词性)]，返回依存树 dict。"""
        pairs = self.tagger.tag(text) if isinstance(text, str) else list(text)
        words = [w for w, _ in pairs]
        tags = [t for _, t in pairs]
        heads, rels = self._parse(words, tags)
        return {
            "words": words,
            "tags": tags,
            "heads": heads,          # heads[i] 为词 i 的支配词下标，根为 -1
            "relations": rels,
        }

    # -- 核心算法 ---------------------------------------------------------
    def _parse(self, words: list[str], tags: list[str]) -> tuple[list[int], list[str]]:
        n = len(words)
        if n == 0:
            return [], []
        root = self._find_root(tags)
        heads: list[Optional[int]] = [None] * n
        rels: list[Optional[str]] = [None] * n
        heads[root] = -1
        rels[root] = "root"

        for i in range(n):
            if i == root:
                continue
            head, rel = self._head_of(words, tags, i, root)
            heads[i] = head
            rels[i] = rel

        return [h if h is not None else root for h in heads], \
               [r if r is not None else "dep" for r in rels]

    @staticmethod
    def _find_root(tags: list[str]) -> int:
        # 主谓句：主句动词为根；无动词则用形容词；否则用第一个实词
        for i, t in enumerate(tags):
            if t == "v":
                return i
        for i, t in enumerate(tags):
            if t == "a":
                return i
        for i, t in enumerate(tags):
            if t in ("n", "r", "ns", "nr", "nt"):
                return i
        return 0

    def _head_of(self, words: list[str], tags: list[str], i: int, root: int):
        """为下标 i 的词确定支配词和关系。"""
        t = tags[i]
        n = len(words)

        if t == "u":  # 助词：依附于前一个词
            return self._safe(i - 1, root, n), "mark"
        if t == "d":  # 副词：依附于后面最近的动词/形容词/名词
            for j in range(i + 1, n):
                if tags[j] in ("v", "a"):
                    return j, "advmod"
            return self._safe(i + 1, root, n), "advmod"
        if t == "p":  # 介词：依附于后面最近的动词或名词
            for j in range(i + 1, n):
                if tags[j] in ("v", "n", "r", "ns", "nt"):
                    return j, "case"
            return self._safe(i + 1, root, n), "case"
        if t == "c":  # 连词：依附于后面最近动词
            for j in range(i + 1, n):
                if tags[j] == "v":
                    return j, "conj"
            return self._safe(i + 1, root, n), "conj"
        if t in ("m", "q"):  # 数词/量词：依附于后面最近的名词
            for j in range(i + 1, n):
                if tags[j] in ("n", "ns", "nt", "nr"):
                    return j, "nummod" if t == "m" else "clf"
            return self._safe(i - 1, root, n), "nummod"
        if t == "a":  # 形容词：后面有名词则作定语，否则依附于根
            for j in range(i + 1, n):
                if tags[j] in ("n", "ns", "nt", "nr"):
                    return j, "amod"
            return root, "dep"
        if t == "f":  # 方位词：依附于前面名词
            for j in range(i - 1, -1, -1):
                if tags[j] in ("n", "ns", "nt", "nr", "f"):
                    return j, "lobj"
            return self._safe(i - 1, root, n), "lobj"
        if t == "t":  # 时间词：依附于根
            return root, "tmod"
        if t in ("r", "n", "ns", "nr", "nt"):
            return self._nominal_head(words, tags, i, root)
        if t == "v":
            # 非根动词：依附于根（并列/补足）
            return root, "conj"
        # 其它：默认依附于前一个词
        return self._safe(i - 1, root, n), "dep"

    def _nominal_head(self, words, tags, i, root):
        """名词性成分的支配词：主语依附于后面的动词，宾语依附于前面的动词。"""
        n = len(tags)
        # 后面有动词 -> 主语
        for j in range(i + 1, n):
            if tags[j] == "v":
                return j, "nsubj"
        # 前面有动词 -> 宾语
        for j in range(i - 1, -1, -1):
            if tags[j] == "v":
                return j, "dobj"
        # 后面有名词 -> 复合/定语
        for j in range(i + 1, n):
            if tags[j] in ("n", "ns", "nt", "nr"):
                return j, "compound"
        return root, "dep"

    @staticmethod
    def _safe(idx, fallback, n):
        if 0 <= idx < n:
            return idx
        return fallback


# ---------------------------------------------------------------------------
# 短语结构分析（CYK）
# ---------------------------------------------------------------------------

# (左部, 右部元组) 规则表；右部为 1 或 2 个符号（标签级 CNF）
_GRAMMAR = [
    # 一元规则：词性 -> 短语
    ("NP", ("n",)), ("NP", ("r",)), ("NP", ("ns",)), ("NP", ("nr",)),
    ("NP", ("nt",)), ("NP", ("t",)), ("NP", ("f",)), ("NP", ("m",)),
    ("NP", ("q",)), ("NP", ("x",)),
    ("VP", ("v",)), ("AP", ("a",)), ("ADVP", ("d",)), ("P", ("p",)),
    ("CONJ", ("c",)), ("U", ("u",)),
    # 二元规则：短语组合
    ("NP", ("a", "n")),          # 红花
    ("NP", ("m", "n")),          # 三个人
    ("NP", ("m", "q")),          # 三个
    ("NP", ("NP", "NP")),        # 名词复合 / 并列
    ("NP", ("NP", "U")),         # X的（的字结构作定语）
    ("NP", ("NP", "PP")),        # 带定语的名词
    ("NP", ("n", "v")),          # 动词作定语（机器学习）
    ("NP", ("v", "n")),          # 动名复合（学习系统）
    ("VP", ("v", "NP")),         # 动宾
    ("VP", ("v", "VP")),         # 动词带补足
    ("VP", ("d", "VP")),         # 副词 + 动词短语
    ("VP", ("ADVP", "VP")),      # 状语 + 动词短语
    ("VP", ("VP", "U")),         # 动词 + 体助词（开发了）
    ("VP", ("VP", "PP")),        # 动词短语 + 介词短语
    ("VP", ("VP", "NP")),        # 双宾语 / 动量
    ("AP", ("d", "AP")),         # 很漂亮
    ("AP", ("ADVP", "AP")),      # 非常漂亮
    ("PP", ("P", "NP")),         # 介词短语
    ("TP", ("NP", "f")),         # 名词 + 方位（时间/空间短语）
    ("S", ("NP", "VP")),         # 主谓
    ("S", ("NP", "AP")),         # 名词 + 形容词谓语
    ("S", ("NP", "TP")),         # 名词 + 方位
    ("S", ("VP",)),              # 无主句
    ("S", ("AP",)),              # 无主形容词句
    ("S", ("S", "PP")),          # 带状语从句
    ("S", ("S", "CONJ")),        # 连词尾
]


class ConstituencyParser:
    """基于 CYK 的短语结构分析器。"""

    def __init__(self):
        self._rule_map: dict = {}
        for lhs, rhs in _GRAMMAR:
            self._rule_map.setdefault(rhs, []).append(lhs)
        # 优先选择更高层的成分
        self._preference = {"S": 0, "VP": 1, "NP": 2, "AP": 2, "TP": 3,
                            "PP": 4, "ADVP": 5, "P": 6, "CONJ": 7, "U": 8}

    def parse(self, text) -> dict:
        from .pos import POSTagger as _T
        pairs = _T().tag(text) if isinstance(text, str) else list(text)
        tags = [t for _, t in pairs]
        words = [w for w, _ in pairs]
        tree = self._cyk(tags, words)
        return {"words": words, "tags": tags, "tree": tree}

    def _cyk(self, tags: list[str], words: list[str]) -> dict:
        n = len(tags)
        if n == 0:
            return {"label": "S", "children": []}

        # table[i][j] -> dict(lhs -> (prob, back))
        table: list[list[dict]] = [[{} for _ in range(n)] for _ in range(n)]

        # 初始化：一元规则
        for i in range(n):
            for lhs in self._rule_map.get((tags[i],), []):
                table[i][i][lhs] = (0.0, None)
            # 任何词性也可直接充当叶子（若没有短语规则覆盖）
            table[i][i].setdefault(tags[i], (0.0, None))

        # 动态规划
        for length in range(2, n + 1):
            for i in range(0, n - length + 1):
                j = i + length - 1
                cell = table[i][j]
                for k in range(i, j):
                    left, right = table[i][k], table[k + 1][j]
                    for l_lhs in left:
                        for r_lhs in right:
                            for lhs in self._rule_map.get((l_lhs, r_lhs), []):
                                prob = left[l_lhs][0] + right[r_lhs][0] + 1.0
                                key = (self._preference.get(lhs, 99), lhs)
                                cur = cell.get(lhs)
                                if cur is None or prob > cur[0]:
                                    cell[lhs] = (prob, (k, l_lhs, r_lhs))

        # 回溯：优先整句 S，否则退化为最宽覆盖
        return self._reconstruct(table, tags, words, 0, n - 1)

    def _reconstruct(self, table, tags, words, i, j) -> dict:
        if i == j:
            cell = table[i][j]
            label = self._best_label(cell, [tags[i]]) if cell else tags[i]
            return {"label": label, "word": words[i], "children": []}

        cell = table[i][j]
        # 无组合规则覆盖该区间：扁平化为并列子节点
        if not cell:
            return {"label": "S", "word": None, "children": [
                self._reconstruct(table, tags, words, k, k)
                for k in range(i, j + 1)]}

        label = self._best_label(cell, ["S", "VP", "NP", "AP"])
        prob, back = cell[label]
        if back is None:
            # 一元规则覆盖（理论上只出现在叶子）
            return {"label": label, "word": None, "children": [
                self._reconstruct(table, tags, words, k, k)
                for k in range(i, j + 1)]}
        k, l_lhs, r_lhs = back
        return {
            "label": label,
            "word": None,
            "children": [
                self._reconstruct(table, tags, words, i, k),
                self._reconstruct(table, tags, words, k + 1, j),
            ],
        }

    def _best_label(self, cell: dict, preferred: list[str]) -> str:
        for p in preferred:
            if p in cell:
                return p
        if cell:
            return min(cell.keys(), key=lambda l: self._preference.get(l, 99))
        return "S"
