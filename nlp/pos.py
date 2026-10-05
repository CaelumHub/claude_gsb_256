"""词性标注器。

采用「词典 + HMM + 后缀规则」三路结合：

1. 内置一个手工标注的小型语料，训练 HMM 的转移概率与发射概率；
2. :data:`nlp.lexicon.POS_DICTIONARY` 中的词直接提供高置信度的候选词性；
3. 未登录词通过后缀 / 结构规则生成候选词性；
4. 最后用 Viterbi 在词性网格上解码，得到全局最优标注序列。

标签集（简化版，面向中文常见用法）：
    名词 n、动词 v、形容词 a、副词 d、代词 r、介词 p、连词 c、助词 u、
    数词 m、量词 q、时间词 t、方位词 f、地名 ns、人名 nr、机构 nt、其他 x
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from .hmm import HMM
from .lexicon import (LOCATIONS, ORGANIZATIONS, PERSONS, POS_DICTIONARY,
                      ORG_SUFFIXES, LOC_SUFFIXES, SURNAMES)
from .segmenter import Segmenter


TAGSET = ["n", "v", "a", "d", "r", "p", "c", "u", "m", "q", "t", "f",
          "ns", "nr", "nt", "x"]

TAG_NAMES = {
    "n": "名词", "v": "动词", "a": "形容词", "d": "副词", "r": "代词",
    "p": "介词", "c": "连词", "u": "助词", "m": "数词", "q": "量词",
    "t": "时间词", "f": "方位词", "ns": "地名", "nr": "人名", "nt": "机构",
    "x": "其他",
}


# 手工标注语料：每句为 (词, 词性) 序列，用于训练 HMM 转移概率。
_TAGGED_CORPUS = [
    [("自然语言", "n"), ("处理", "v"), ("是", "v"), ("人工智能", "n"),
     ("的", "u"), ("重要", "a"), ("分支", "n")],
    [("我", "r"), ("喜欢", "v"), ("自然语言", "n"), ("处理", "v")],
    [("北京", "ns"), ("是", "v"), ("中国", "ns"), ("的", "u"), ("首都", "n")],
    [("今天", "t"), ("天气", "n"), ("非常", "d"), ("好", "a")],
    [("他", "r"), ("在", "p"), ("公司", "n"), ("工作", "v")],
    [("我们", "r"), ("正在", "d"), ("开发", "v"), ("一个", "m"), ("系统", "n")],
    [("这个", "r"), ("算法", "n"), ("的", "u"), ("性能", "n"), ("很", "d"),
     ("优秀", "a")],
    [("你", "r"), ("应该", "v"), ("学习", "v"), ("编程", "n")],
    [("人工智能", "n"), ("技术", "n"), ("发展", "v"), ("迅速", "a")],
    [("上海", "ns"), ("的", "u"), ("经济", "n"), ("十分", "d"), ("发达", "a")],
    [("她", "r"), ("非常", "d"), ("喜欢", "v"), ("阅读", "v")],
    [("我们", "r"), ("需要", "v"), ("提高", "v"), ("系统", "n"), ("效率", "n")],
    [("苹果", "n"), ("公司", "n"), ("发布", "v"), ("了", "u"), ("新", "a"),
     ("产品", "n")],
    [("研究", "n"), ("表明", "v"), ("数据", "n"), ("分析", "n"), ("很", "d"),
     ("重要", "a")],
    [("他们", "r"), ("通过", "p"), ("网络", "n"), ("传输", "v"), ("数据", "n")],
    [("小明", "nr"), ("在", "p"), ("学校", "n"), ("学习", "v")],
    [("政府", "n"), ("出台", "v"), ("了", "u"), ("新", "a"), ("政策", "n")],
    [("这个", "r"), ("问题", "n"), ("非常", "d"), ("复杂", "a")],
    [("科学家", "n"), ("发现", "v"), ("了", "u"), ("新", "a"), ("物种", "n")],
    [("市场", "n"), ("需求", "n"), ("持续", "d"), ("增长", "v")],
    [("我们", "r"), ("应该", "v"), ("保护", "v"), ("环境", "n")],
    [("他", "r"), ("成功", "a"), ("地", "u"), ("完成", "v"), ("了", "u"),
     ("任务", "n")],
    [("中国", "ns"), ("经济", "n"), ("保持", "v"), ("稳定", "a"), ("增长", "v")],
    [("老师", "n"), ("耐心", "a"), ("地", "u"), ("讲解", "v"), ("知识", "n")],
    [("机器", "n"), ("学习", "v"), ("是", "v"), ("当前", "t"), ("热门", "a"),
     ("方向", "n")],
    [("我", "r"), ("认为", "v"), ("这个", "r"), ("方案", "n"), ("可行", "a")],
    [("公司", "n"), ("员工", "n"), ("数量", "n"), ("不断", "d"), ("增加", "v")],
    [("他", "r"), ("在", "p"), ("北京", "ns"), ("工作", "v"), ("三年", "m")],
    [("语言", "n"), ("模型", "n"), ("取得", "v"), ("重大", "a"), ("突破", "n")],
    [("我们", "r"), ("需要", "v"), ("更加", "d"), ("高效", "a"), ("的", "u"),
     ("方法", "n")],
    [("医院", "n"), ("引进", "v"), ("了", "u"), ("先进", "a"), ("设备", "n")],
    [("价格", "n"), ("上涨", "v"), ("了", "u"), ("百分之", "m"), ("十", "m")],
    [("她", "r"), ("是", "v"), ("一名", "m"), ("优秀", "a"), ("的", "u"),
     ("工程师", "n")],
    [("春节", "t"), ("期间", "f"), ("人们", "n"), ("返乡", "v"), ("团圆", "v")],
    [("这份", "r"), ("报告", "n"), ("详细", "a"), ("地", "u"), ("分析", "v"),
     ("了", "u"), ("市场", "n")],
    [("人工智能", "n"), ("将", "d"), ("深刻", "d"), ("改变", "v"), ("人类", "n"),
     ("社会", "n")],
    [("我们", "r"), ("坚信", "v"), ("未来", "t"), ("会", "v"), ("更加", "d"),
     ("美好", "a")],
]


class POSTagger:
    def __init__(self, segmenter: Optional[Segmenter] = None):
        self.segmenter = segmenter or Segmenter()
        self.hmm = self._train()
        self.pos_dict = dict(POS_DICTIONARY)
        self._suffix_rules = self._build_suffix_rules()

    # -- 训练 -------------------------------------------------------------
    def _train(self) -> HMM:
        hmm = HMM(TAGSET, add_k=0.01)
        sequences = _TAGGED_CORPUS + self._dict_sequences()
        hmm.train(sequences)
        return hmm

    def _dict_sequences(self) -> list[list[tuple]]:
        """把词典中的词性当作额外训练信号，加权提升覆盖率。"""
        seqs: list[list[tuple]] = []
        for word, tag in POS_DICTIONARY.items():
            if tag in TAGSET:
                seqs.append([(word, tag)])
        return seqs

    @staticmethod
    def _build_suffix_rules() -> list[tuple[str, list[str]]]:
        """后缀 -> 候选词性。"""
        return [
            ("们", ["n", "r"]),
            ("者", ["n"]),
            ("员", ["n"]),
            ("家", ["n"]),
            ("性", ["n", "a"]),
            ("度", ["n"]),
            ("率", ["n"]),
            ("量", ["n"]),
            ("子", ["n"]),
            ("头", ["n"]),
            ("化", ["v", "n"]),
            ("了", ["u", "v"]),
            ("着", ["u", "v"]),
            ("过", ["u", "v"]),
            ("的", ["u", "a"]),
            ("地", ["u", "d"]),
            ("得", ["u", "v"]),
            ("上", ["f", "n"]),
            ("下", ["f", "n"]),
            ("中", ["f", "n"]),
            ("里", ["f", "n"]),
            ("内", ["f", "n"]),
            ("外", ["f", "n"]),
            ("前", ["f", "t"]),
            ("后", ["f", "t"]),
        ]

    # -- 对外接口 ---------------------------------------------------------
    def tag(self, text) -> list[tuple[str, str]]:
        """返回 ``[(词, 词性), ...]``。``text`` 可为句子字符串或词列表。"""
        if isinstance(text, str):
            words = self.segmenter.cut(text)
        else:
            words = list(text)
        return self._tag_words(words)

    def tag_sentences(self, sentences: Iterable[str]):
        return [self.tag(s) for s in sentences]

    # -- 标注 -------------------------------------------------------------
    def _candidate_tags(self, word: str) -> list[str]:
        if word in self.pos_dict:
            return [self.pos_dict[word]]
        # 纯标点/符号 -> 其他
        if not any(c.isalnum() or "一" <= c <= "鿿" for c in word):
            return ["x"]
        # 已知实体词典 -> 对应专名标签
        if word in LOCATIONS:
            return ["ns"]
        if word in ORGANIZATIONS:
            return ["nt"]
        if word in PERSONS:
            return ["nr"]
        if re.fullmatch(r"[0-9]+(?:[.][0-9]+)?%?", word):
            return ["m"]
        if re.fullmatch(r"[0-9]+年|[0-9]+月|[0-9]+日", word):
            return ["t"]
        if re.search(r"[0-9]", word):
            return ["m", "n"]
        # 地名 / 机构 / 人名后缀
        if any(word.endswith(s) for s in LOC_SUFFIXES):
            return ["ns", "n"]
        if any(word.endswith(s) for s in ORG_SUFFIXES):
            return ["nt", "n"]
        if len(word) <= 3 and word and word[0] in SURNAMES:
            return ["nr", "n"]
        for suffix, tags in self._suffix_rules:
            if word.endswith(suffix):
                return tags
        # 默认
        return ["n", "v", "a"]

    def _tag_words(self, words: list[str]) -> list[tuple[str, str]]:
        if not words:
            return []
        n = len(words)
        k = len(TAGSET)
        neg = float("-inf")

        # 每个词 -> 候选词性（未登录词用规则，登录词用词典）
        candidates: list[list[str]] = []
        for w in words:
            tags = self._candidate_tags(w)
            candidates.append([t for t in tags if t in TAGSET] or ["x"])

        viterbi = [[neg] * k for _ in range(n)]
        backptr = [[-1] * k for _ in range(n)]

        for s, tag in enumerate(TAGSET):
            if tag in candidates[0]:
                viterbi[0][s] = (self.hmm.start_logp(tag)
                                 + self.hmm.emit_logp(words[0], tag))
            else:
                viterbi[0][s] = neg

        for t in range(1, n):
            for s, tag in enumerate(TAGSET):
                if tag not in candidates[t]:
                    viterbi[t][s] = neg
                    continue
                best = neg
                best_prev = -1
                for p in range(k):
                    score = (viterbi[t - 1][p]
                             + self.hmm.trans_logp(TAGSET[p], tag))
                    if score > best:
                        best = score
                        best_prev = p
                viterbi[t][s] = best + self.hmm.emit_logp(words[t], tag)
                backptr[t][s] = best_prev

        # 处理完全无合法路径的情况
        last = max(range(k), key=lambda s: viterbi[n - 1][s])
        if viterbi[n - 1][last] == neg:
            # 退化为逐词规则标注
            return [(w, candidates[i][0]) for i, w in enumerate(words)]

        path = [TAGSET[last]]
        for t in range(n - 1, 0, -1):
            last = backptr[t][last]
            if last < 0:
                break
            path.append(TAGSET[last])
        path.reverse()
        if len(path) < n:
            path = [candidates[i][0] for i in range(n - len(path))] + path
        return list(zip(words, path))
