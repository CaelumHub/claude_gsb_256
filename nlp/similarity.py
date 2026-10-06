"""文本相似度与近重复检测（查重）的核心算法。

难点定位
--------
语料库中大量「同一段话的翻版」——同义改写、语序调整、局部摘录——
需要一套**确定、可复现、与语料规模无关**的相似度口径，以及把
全库两两比较的 O(n²) 代价降到可接受范围的候选对生成机制。

设计要点
--------
1. **双通道特征**
   - 字符 n-gram（默认 3-gram）：对语序微调、局部措辞改动鲁棒；
     「主题相近但内容不同」的文档在字面上重合度很低，可有效抑制误判。
   - 同义词归一化的内容词集合：分词 → 去停用词 → 查
     :data:`nlp.lexicon.SYNONYM_MAP` 归一到规范词，吸收同义替换。
2. **相似度 = Jaccard + 包含度（containment）加权融合**
   - Jaccard 衡量整体一致性；包含度 ``|A∩B| / min(|A|,|B|)`` 对
     「一篇是另一篇的截取/摘录」敏感（长文包含短文时 Jaccard 会被稀释）。
   - 相似度只依赖两篇文档本身（不引入随语料变化的 IDF 等统计量），
     因此**同一对文档无论何时评估，得分完全一致**（口径前后一致）。
3. **候选对生成：one-permutation MinHash + 分带（banding）**
   - 每篇文档用 R 个独立哈希把特征集合压缩成定长签名，
     只需 O(特征数) 次哈希（传统 k 置换 MinHash 需要 O(k·特征数)）；
   - 签名切成若干「带」，任一带完全相同即成为候选对，再做精确复核；
   - 哈希用 blake2b（进程间稳定，Python 内置 ``hash()`` 有随机盐，不可用）。
4. **归组：星形（质心）贪心聚类**
   - 组内每个成员都与组中心高度相似（≥ 阈值），避免单链聚类把
     「主题相近但内容不同」的文档沿传递边串成一组；
   - 边按分数从高到低贪心处理，输出确定、可复现。

复杂度：建索引 O(Σ特征数)；查重候选对生成 O(n·带数)，精确复核只发生在
候选对上；n≤ALL_PAIRS_LIMIT 时直接全配对精确比较（小库零近似误差）。
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, Iterator, Optional

from .lexicon import STOPWORDS, SYNONYM_MAP, canonical_word
from .segmenter import Segmenter


# ---------------------------------------------------------------------------
# 参数（均为口径的一部分，改动会改变历史得分，应谨慎）
# ---------------------------------------------------------------------------

#: 字符 n-gram 长度
SHINGLE_SIZE = 3

#: 相似度各分量权重（和为 1）
WEIGHT_SHINGLE_JACCARD = 0.25
WEIGHT_SHINGLE_CONTAINMENT = 0.35
WEIGHT_TOKEN_JACCARD = 0.15
WEIGHT_TOKEN_CONTAINMENT = 0.25

#: 判定「高度相似 / 重复」的默认阈值
DEFAULT_THRESHOLD = 0.6

#: MinHash 签名：R 个独立哈希 × 每个哈希 BINS 个桶
SIG_HASHES = 6
SIG_BINS = 64
#: 分带时每带的行数（带数 = R * BINS / BAND_ROWS = 96）
BAND_ROWS = 4

#: 文档数不超过该值时跳过 LSH，直接全配对精确比较
ALL_PAIRS_LIMIT = 150

_WORD_CHAR_RE = re.compile(r"[0-9a-zA-Z一-鿿]")
_KEEP_RE = re.compile(r"[0-9a-zA-Z一-鿿]+")

#: 相似度视角下的额外功能词（代词等，不携带内容区分度）
_EXTRA_STOPWORDS = {
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们", "它们",
    "咱", "咱们", "俺", "您", "这人", "那人",
}

#: 同义词键的最长字符数（跨 token 合并匹配的上界）
_MAX_SYNONYM_CHARS = max(len(k) for k in SYNONYM_MAP) if SYNONYM_MAP else 1


def id_sort_key(record_id: str):
    """语料 id 的确定性排序键：``corpus_2`` 排在 ``corpus_10`` 前。

    用于全局排序的决胜键（相似度相同按 id 升序），保证翻页时
    顺序连续、不重不漏。
    """
    match = re.search(r"(\d+)$", record_id or "")
    if match:
        return (0, int(match.group(1)), record_id)
    return (1, 0, record_id or "")


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """规范化文本：小写、去除空白与标点，只保留字母/数字/汉字。

    字符级特征基于该结果提取，因此对排版、空白、标点差异免疫。
    """
    return "".join(_KEEP_RE.findall((text or "").lower()))


def content_tokens(text: str, segmenter: Optional[Segmenter] = None) -> list[str]:
    """提取内容词：分词 → 去功能词 → 同义词归一化。

    同义词匹配支持**跨 token 合并**：分词器未收录的同义词变体
    （如「因特网」被切成「因特 / 网」）会先按最长匹配拼回再归一，
    保证改写前后的词集合一致。
    """
    seg = segmenter or Segmenter()
    raw = [w.lower() for w in seg.cut(text or "")]
    raw = [w for w in raw if _WORD_CHAR_RE.search(w)]

    tokens = []
    i = 0
    while i < len(raw):
        # 最长匹配：raw[i:j] 拼接后命中同义词键则合并
        piece, end = None, i + 1
        joined = ""
        for j in range(i, min(len(raw), i + _MAX_SYNONYM_CHARS)):
            joined += raw[j]
            if len(joined) > _MAX_SYNONYM_CHARS:
                break
            if joined in SYNONYM_MAP and len(joined) > 1:
                piece, end = joined, j + 1
        word = piece if piece is not None else raw[i]
        if word not in STOPWORDS and word not in _EXTRA_STOPWORDS:
            tokens.append(canonical_word(word))
        i = end
    return tokens


def char_shingles(normalized: str, n: int = SHINGLE_SIZE) -> set[str]:
    """字符 n-gram 集合；文本短于 n 时整体作为唯一 shingle。"""
    if not normalized:
        return set()
    if len(normalized) <= n:
        return {normalized}
    return {normalized[i:i + n] for i in range(len(normalized) - n + 1)}


def _hash64(value: str, seed: int) -> int:
    """进程间稳定的 64 位哈希（blake2b，person 字段区分不同哈希函数）。"""
    person = f"nlp-sim-{seed}".encode("ascii")[:16]
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8,
                             person=person).digest()
    return int.from_bytes(digest, "little")


def minhash_signature(features: Iterable[str],
                      hashes: int = SIG_HASHES,
                      bins: int = SIG_BINS) -> list[list[Optional[int]]]:
    """one-permutation MinHash 签名（带 densification）。

    对每个哈希函数：把 64 位哈希值的高若干位作为桶号，桶内取最小值。
    两篇文档同一位置签名值相等的概率 ≈ 特征集合的 Jaccard，
    因此签名可直接用于分带候选对生成。

    特征数少于桶数时会出现空桶，空桶按「向右循环借最近非空桶的值」
    致密化（densified OPH），避免短文档因空桶丢失候选召回。
    返回 ``hashes × bins`` 的嵌套列表；完全无特征时桶为 ``None``。
    """
    shift = 64 - (bins - 1).bit_length()
    signature: list[list[Optional[int]]] = [[None] * bins for _ in range(hashes)]
    for feature in features:
        for r in range(hashes):
            h = _hash64(feature, r)
            bucket = h >> shift
            low = h & ((1 << shift) - 1)
            current = signature[r][bucket]
            if current is None or low < current:
                signature[r][bucket] = low

    # 致密化：空桶借用右侧最近非空桶的值（确定性）
    for row in signature:
        non_empty = [i for i, v in enumerate(row) if v is not None]
        if not non_empty or len(non_empty) == bins:
            continue
        for i, v in enumerate(row):
            if v is not None:
                continue
            for step in range(1, bins + 1):
                donor = row[(i + step) % bins]
                if donor is not None:
                    row[i] = donor
                    break
    return signature


class DocFeatures:
    """一篇文档的相似度特征（可 JSON 序列化为索引记录）。"""

    __slots__ = ("tokens", "shingles", "signature")

    def __init__(self, tokens: set, shingles: set, signature):
        self.tokens = tokens
        self.shingles = shingles
        self.signature = signature

    # -- 序列化 -----------------------------------------------------------
    def to_record(self) -> dict:
        return {
            "tokens": sorted(self.tokens),
            "shingles": sorted(self.shingles),
            "signature": [list(row) for row in self.signature],
        }

    @classmethod
    def from_record(cls, record: dict) -> "DocFeatures":
        return cls(
            tokens=set(record.get("tokens") or []),
            shingles=set(record.get("shingles") or []),
            signature=[list(row) for row in record.get("signature") or []],
        )


def extract_features(text: str,
                     segmenter: Optional[Segmenter] = None) -> DocFeatures:
    """从原始文本提取全部相似度特征。"""
    seg = segmenter or Segmenter()
    tokens = set(content_tokens(text, seg))
    shingles = char_shingles(normalize_text(text))
    # 签名在「词 + 字 shingle」的并集上计算：词级特征把同义归一化后的
    # 重合也反映进签名，提高改写文档的候选召回。
    combined = {f"w:{t}" for t in tokens} | {f"s:{s}" for s in shingles}
    signature = minhash_signature(combined)
    return DocFeatures(tokens=tokens, shingles=shingles, signature=signature)


# ---------------------------------------------------------------------------
# 相似度
# ---------------------------------------------------------------------------

def _jaccard(a: set, b: set) -> float:
    union = len(a) + len(b)
    if union == 0:
        return 0.0
    inter = len(a & b)
    union -= inter
    return inter / union if union else 0.0


def _containment(a: set, b: set) -> float:
    smaller = min(len(a), len(b))
    if smaller == 0:
        return 0.0
    return len(a & b) / smaller


def score_features(a: DocFeatures, b: DocFeatures) -> dict:
    """计算两篇文档的相似度及各分量（确定性、对称）。"""
    s_jac = _jaccard(a.shingles, b.shingles)
    s_con = _containment(a.shingles, b.shingles)
    t_jac = _jaccard(a.tokens, b.tokens)
    t_con = _containment(a.tokens, b.tokens)
    score = (WEIGHT_SHINGLE_JACCARD * s_jac +
             WEIGHT_SHINGLE_CONTAINMENT * s_con +
             WEIGHT_TOKEN_JACCARD * t_jac +
             WEIGHT_TOKEN_CONTAINMENT * t_con)
    return {
        "score": score,
        "shingle_jaccard": s_jac,
        "shingle_containment": s_con,
        "token_jaccard": t_jac,
        "token_containment": t_con,
    }


# ---------------------------------------------------------------------------
# 候选对生成（LSH 分带）
# ---------------------------------------------------------------------------

def lsh_band_keys(signature, rows: int = BAND_ROWS) -> Iterator[tuple]:
    """把签名切成若干带，产出每个带的键；含空桶的带不产键。"""
    for r, row in enumerate(signature):
        for start in range(0, len(row), rows):
            band = row[start:start + rows]
            if len(band) < rows or any(v is None for v in band):
                continue
            yield (r, start, tuple(band))


def candidate_pairs(signatures: dict,
                    rows: int = BAND_ROWS) -> set[tuple]:
    """从签名中找出候选文档对（任一带相同即候选）。

    返回 ``{(id_a, id_b), ...}``，其中 ``id_a < id_b``（按 id_sort_key）。
    """
    buckets: dict[tuple, list] = {}
    for doc_id in sorted(signatures, key=id_sort_key):
        for key in lsh_band_keys(signatures[doc_id], rows):
            buckets.setdefault(key, []).append(doc_id)

    pairs: set[tuple] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    return pairs


def all_pairs(ids: Iterable[str]) -> set[tuple]:
    """全配对（小语料库用，零近似误差）。"""
    ordered = sorted(ids, key=id_sort_key)
    return {(ordered[i], ordered[j])
            for i in range(len(ordered)) for j in range(i + 1, len(ordered))}


# ---------------------------------------------------------------------------
# 星形（质心）贪心聚类
# ---------------------------------------------------------------------------

def cluster(edges: Iterable[tuple]) -> list[tuple]:
    """把「相似」文档边归成星形组，返回 ``[(center, members), ...]``。

    不用单链连通分量：单链会沿传递边把「主题相近但内容不同」的文档
    串成一组（A~B、B~C 相似但 A≁C，单链仍把三者归一组）。
    星形聚类保证**组内每个成员都与组中心高度相似（≥ 阈值）**：

    1. 边按分数从高到低处理（同分按 id 排序，确定性）；
    2. 两端都未入组 → 以度数高者（并列取 id 小者）为中心建新组；
    3. 一端未入组 → 仅当它与**组中心**的相似度达标时才加入；
    4. 两端都已入组 → 跳过，既有归组不被合并冲乱。

    同一篇原文的不同摘录互不相似、但都与原文相似时，会正确地
    聚到以原文为中心的一组。返回的 members 按 id_sort_key 排序，
    组间按最小组内 id 排序（输出确定）。
    """
    edges = [(a, b, s) for a, b, s in edges]
    adjacency: dict[str, dict[str, float]] = {}
    for a, b, s in edges:
        adjacency.setdefault(a, {})[b] = s
        adjacency.setdefault(b, {})[a] = s
    degree = {doc: len(peers) for doc, peers in adjacency.items()}

    ordered = sorted(edges, key=lambda e: (-e[2], id_sort_key(e[0]),
                                           id_sort_key(e[1])))
    center_of: dict[str, str] = {}
    members: dict[str, list[str]] = {}

    for a, b, _ in ordered:
        ca, cb = center_of.get(a), center_of.get(b)
        if ca is None and cb is None:
            # 新组：度数高者为中心，并列取 id 小者
            if degree[a] != degree[b]:
                center = a if degree[a] > degree[b] else b
            else:
                center = a if id_sort_key(a) <= id_sort_key(b) else b
            other = b if center == a else a
            center_of[center] = center
            center_of[other] = center
            members[center] = [center, other]
        elif ca is not None and cb is None:
            if ca in adjacency.get(b, {}):
                center_of[b] = ca
                members[ca].append(b)
        elif cb is not None and ca is None:
            if cb in adjacency.get(a, {}):
                center_of[a] = cb
                members[cb].append(a)
        # 两端都已入组：跳过，不合并既有组

    groups = [(center, sorted(mems, key=id_sort_key))
              for center, mems in members.items() if len(mems) >= 2]
    groups.sort(key=lambda item: id_sort_key(item[1][0]))
    return groups
