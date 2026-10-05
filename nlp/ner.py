"""命名实体识别（NER）。

三路结合：
1. **字典匹配**：用内置实体词典（人名 / 地名 / 机构）做最长匹配；
2. **正则规则**：识别时间、日期、数字、金额、百分比；
3. **结构规则**：姓氏 + 名字组合识别人名，后缀识别机构 / 地名。

实体类别：PERSON 人名、LOCATION 地名、ORGANIZATION 机构、TIME 时间、
DATE 日期、NUMBER 数字、MONEY 金额、PERCENT 百分比。
"""

from __future__ import annotations

import re
from typing import Optional

from .lexicon import (LOCATIONS, ORGANIZATIONS, PERSONS, SURNAMES,
                      ORG_SUFFIXES, LOC_SUFFIXES, GIVEN_NAME_CHARS, load_dictionary)
from .segmenter import Segmenter


ENTITY_TYPE_NAMES = {
    "PERSON": "人名", "LOCATION": "地名", "ORGANIZATION": "机构",
    "TIME": "时间", "DATE": "日期", "NUMBER": "数字", "MONEY": "金额",
    "PERCENT": "百分比",
}


# 正则规则：按优先级排列
_REGEX_RULES = [
    ("DATE", re.compile(r"\d{4}[年/\-]\d{1,2}[月/\-]\d{1,2}日?")),
    ("DATE", re.compile(r"\d{1,2}月\d{1,2}日")),
    ("TIME", re.compile(r"\d{1,2}[:：]\d{2}(?:[:：]\d{2})?")),
    ("MONEY", re.compile(r"\d+(?:\.\d+)?(?:万元|亿元|人民币|美元|港元|港币|欧元|日元|英镑|元)")),
    ("PERCENT", re.compile(r"百分之[零一二三四五六七八九十百]+|\d+(?:\.\d+)?%")),
    ("NUMBER", re.compile(r"\d+(?:\.\d+)?")),
]


class NERExtractor:
    def __init__(self, segmenter: Optional[Segmenter] = None):
        self.segmenter = segmenter or Segmenter()
        self.dictionary = load_dictionary()
        # 合并实体词典：实体名 -> 类型
        self._entity_dict: dict[str, str] = {}
        for name in PERSONS:
            self._entity_dict[name] = "PERSON"
        for name in LOCATIONS:
            self._entity_dict[name] = "LOCATION"
        for name in ORGANIZATIONS:
            self._entity_dict[name] = "ORGANIZATION"
        self._max_entity_len = max((len(k) for k in self._entity_dict), default=6)

    # -- 对外接口 ---------------------------------------------------------
    def recognize(self, text: str) -> list[dict]:
        """返回实体列表，每个实体含 ``start/end/text/type``（字符偏移）。"""
        entities: list[dict] = []
        consumed: list[tuple[int, int]] = []

        # 1. 正则（时间/日期/数字/金额/百分比）
        for etype, pattern in _REGEX_RULES:
            for m in pattern.finditer(text):
                span = (m.start(), m.end())
                if self._overlaps(span, consumed):
                    continue
                consumed.append(span)
                entities.append({
                    "start": m.start(), "end": m.end(),
                    "text": m.group(), "type": etype,
                })

        # 2. 字典最长匹配（人名/地名/机构）
        dict_entities = self._dict_match(text)
        for ent in dict_entities:
            span = (ent["start"], ent["end"])
            if self._overlaps(span, consumed):
                continue
            consumed.append(span)
            entities.append(ent)

        # 3. 结构规则（姓氏+名 / 后缀）
        rule_entities = self._rule_match(text)
        for ent in rule_entities:
            span = (ent["start"], ent["end"])
            if self._overlaps(span, consumed):
                continue
            consumed.append(span)
            entities.append(ent)

        entities.sort(key=lambda e: e["start"])
        return entities

    # -- 字典匹配 ---------------------------------------------------------
    def _dict_match(self, text: str) -> list[dict]:
        results = []
        n = len(text)
        i = 0
        while i < n:
            if not ("一" <= text[i] <= "鿿"):
                i += 1
                continue
            matched = None
            for length in range(min(self._max_entity_len, n - i), 0, -1):
                word = text[i:i + length]
                if word in self._entity_dict:
                    matched = (word, self._entity_dict[word], length)
                    break
            if matched:
                word, etype, length = matched
                results.append({
                    "start": i, "end": i + length, "text": word, "type": etype,
                })
                i += length
            else:
                i += 1
        return results

    # -- 结构规则 ---------------------------------------------------------
    def _rule_match(self, text: str) -> list[dict]:
        results: list[dict] = []
        tokens = self._tokenize_with_offsets(text)
        # 常见称谓后缀，用于剔除「王先生」这类误报
        title_suffixes = ("先生", "女士", "小姐", "同志", "老师", "教授",
                          "博士", "经理", "局长", "主席", "书记")

        for word, start, end in tokens:
            if end - start < 2:
                continue
            if not all("一" <= c <= "鿿" for c in word):
                continue
            matched = False
            # 机构后缀
            for suffix in sorted(ORG_SUFFIXES, key=len, reverse=True):
                if word.endswith(suffix) and len(word) >= len(suffix) + 1:
                    results.append({"start": start, "end": end,
                                    "text": word, "type": "ORGANIZATION"})
                    matched = True
                    break
            if matched:
                continue
            # 地名后缀
            for suffix in sorted(LOC_SUFFIXES, key=len, reverse=True):
                if word.endswith(suffix) and len(word) >= len(suffix) + 1:
                    results.append({"start": start, "end": end,
                                    "text": word, "type": "LOCATION"})
                    matched = True
                    break
            if matched:
                continue
            # 人名：整词 2~3 字、以姓氏开头、且不是词典词（避免误报）
            if (2 <= len(word) <= 3 and word[0] in SURNAMES
                    and word not in self.dictionary
                    and not word.endswith(title_suffixes)
                    and (len(word) == 2 or word[1] in GIVEN_NAME_CHARS)):
                results.append({"start": start, "end": end,
                                "text": word, "type": "PERSON"})

        return results

    # -- 工具 -------------------------------------------------------------
    def _tokenize_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        words = self.segmenter.cut(text)
        tokens = []
        pos = 0
        for word in words:
            idx = text.find(word, pos)
            if idx < 0:
                idx = pos
            tokens.append((word, idx, idx + len(word)))
            pos = idx + len(word)
        return tokens

    @staticmethod
    def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
        s, e = span
        for (a, b) in spans:
            if s < b and a < e:
                return True
        return False


def annotate(text: str, entities: list[dict]) -> str:
    """把实体标注回文本，用 XML 风格标签包裹（用于导出/展示）。"""
    parts = []
    last = 0
    for ent in sorted(entities, key=lambda e: e["start"]):
        parts.append(text[last:ent["start"]])
        parts.append(f"<{ent['type']}>{text[ent['start']:ent['end']]}</{ent['type']}>")
        last = ent["end"]
    parts.append(text[last:])
    return "".join(parts)
