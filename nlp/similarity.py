"""文本相似度与近重复检测（纯 Python 实现）。

需求与设计
----------
语料里很多文档是同一段话的翻版：改措辞、调语序、截取片段。本模块提供两类能力：

1. **相似检索**：对任意一段文本，给出与全库文档的相似度并稳定排序、分页；
2. **近重复查重**：扫描全库，把高度相似的文档用并查集归成一组。

为了同时做到「容忍同义替换/语序调整」与「不把同主题不同内容误判为重复」，
指纹由**两个互补通道**组成：

* 词法通道（三个粒度融合）：
    - 内容词 TF-IDF 余弦（对增删少量句子稳健）；
    - **字一元 TF-IDF 余弦**：中文里同义替换往往保留部分汉字
      （屏幕/显示屏、相机/拍照），字粒度天然吸收这类替换，且与
      「同主题不同内容」有明显分差；
    - 全文汉字 bigram 集合的 Jaccard：调换段落/句子顺序只损失句界处的
      个别 bigram，对语序调整近乎免疫；
    - 对 ``{词, 字bigram}`` 混合 shingle 做 **MinHash + LSH**，用于近线性
      候选召回。
* 语义通道：
    - 在全库上统计词共现，做稀疏 PPMI，得到每个词的「上下文分布向量」，
      文档向量 = 词的二阶上下文向量按 TF-IDF 加权求和后归一化。共享上下文
      的同义词因此能产生向量重叠，而不必依赖稠密 SVD（纯 Python 下保持
      可接受的构建耗时）；
    - 对稀疏语义向量做 **随机超平面 LSH**（特征哈希生成超平面，确定性），
      召回措辞差异大但语义接近的候选对。

最终相似度::

    lexical = 0.45 * 词余弦 + 0.30 * 字一元余弦 + 0.25 * 字bigram Jaccard
    score   = sem_weight * 语义余弦 + (1 - sem_weight) * lexical

排序/检索直接使用 ``score``；**判定重复**时除总分阈值外还要求
``lexical >= lex_floor``，保留一份「共同表面文本」证据，避免主题相同、
内容不同的两篇（词法几乎不重叠）仅凭语义被误归为一组。阈值全部可调。

所有随机量（MinHash 置换、超平面哈希）都由固定种子派生，指纹方案带版本号，
保证同一份语料任意时刻重建得到完全一致的分数（口径前后一致）。
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from typing import Optional

from .segmenter import Segmenter
from .text import filter_stopwords

# 指纹/打分口径版本。修改指纹或评分公式时必须 +1，
# 旧索引检测到版本不一致会触发全量重建。
FP_VERSION = 1

# 影响指纹内容的默认参数（阈值类参数不在这里，调阈值无需重建索引）
DEFAULT_CONFIG = {
    "fp_version": FP_VERSION,
    "minhash_perm": 48,      # MinHash 置换数（= bands*rows）
    "lex_bands": 48,         # 词法 LSH：48×1，S 曲线拐点约在 J≈0.62
    "lex_rows": 1,
    "sem_bits": 64,          # 随机超平面位数
    "sem_bands": 32,         # 语义 LSH：32×2，拐点约在 cos≈0.54
    "sem_rows": 2,
    "window": 5,             # 共现窗口
    "context_topk": 10,      # 每个词保留的 PPMI 上下文数
    "vocab_size": 8000,      # 参与共现统计的高频词上限
    "sem_weight": 0.45,      # 总分中语义通道权重
    "seed": 20240601,
}

# 小库直接全对比较，省去 LSH 的漏检风险
BRUTE_FORCE_LIMIT = 200

_MERSENNE_61 = (1 << 61) - 1
_MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# 分词与表面特征
# ---------------------------------------------------------------------------

def content_tokens(text: str, segmenter: Optional[Segmenter] = None) -> list[str]:
    """切词并保留有实义的内容词。

    去掉停用词、标点、空白；英文/数字词保留长度 >= 2 的。
    """
    seg = segmenter or _DEFAULT_SEG
    words = seg.cut(text or "")
    return filter_stopwords(words)


def _cjk_chars(text: str) -> list[str]:
    return [c for c in (text or "") if "一" <= c <= "鿿"]


def char_unigrams(text: str) -> Counter:
    return Counter(_cjk_chars(text))


def char_bigrams(text: str) -> set[str]:
    """全文汉字 bigram 集合（标点/非汉字自然断开，不跨标点配对）。"""
    grams: set[str] = set()
    chars = _cjk_chars(text)
    # 仅统计在原文中相邻的汉字对（_cjk_chars 会跳过标点造成的邻接）
    prev = None
    for i, c in enumerate(text or ""):
        if "一" <= c <= "鿿":
            if prev is not None and i - prev[0] == 1:
                grams.add(prev[1] + c)
            prev = (i, c)
        else:
            prev = None
    # 单字文档兜底
    if not grams and len(chars) == 1:
        grams.add(chars[0])
    return grams


def shingle_set(tokens: list[str], bigrams: set[str]) -> set[str]:
    """MinHash 用的混合 shingle：词一元 + 字 bigram。"""
    shingles: set[str] = set()
    for w in tokens:
        shingles.add("w:" + w)
    for g in bigrams:
        shingles.add("g:" + g)
    return shingles


_DEFAULT_SEG = Segmenter()


# ---------------------------------------------------------------------------
# MinHash（词法通道 LSH）
# ---------------------------------------------------------------------------

class MinHasher:
    """固定置换的 MinHash。

    每条 shingle 取两个 64 位哈希作为基底，置换 i 上的最小值为
    ``(a_i * h + b_i) mod p``（p = 2^61-1）。置换参数由种子确定性派生。
    """

    def __init__(self, num_perm: int = 128, seed: int = DEFAULT_CONFIG["seed"]):
        self.num_perm = num_perm
        rng = random.Random(seed)
        self._a = [rng.randrange(1, _MERSENNE_61) for _ in range(num_perm)]
        self._b = [rng.randrange(0, _MERSENNE_61) for _ in range(num_perm)]

    @staticmethod
    def _hash64(shingle: str) -> int:
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") & _MASK64

    def signature(self, shingles: set[str]) -> list[int]:
        if not shingles:
            return [0] * self.num_perm
        vec = [_MASK64] * self.num_perm
        for shingle in shingles:
            h = self._hash64(shingle)
            for i in range(self.num_perm):
                v = (self._a[i] * h + self._b[i]) % _MERSENNE_61
                if v < vec[i]:
                    vec[i] = v
        return vec


# ---------------------------------------------------------------------------
# 随机超平面签名（语义通道 LSH，稀疏向量友好）
# ---------------------------------------------------------------------------

def hyperplane_signature(vector: dict[str, float], bits: int,
                         seed: int = DEFAULT_CONFIG["seed"]) -> bytes:
    """对稀疏向量求 ``bits`` 个随机超平面的符号位。

    第 j 个超平面在特征 f 上的分量由 ``hash(f, j)`` 确定性地取 ±1，
    无需物化稠密超平面，复杂度 O(nnz × bits)。
    """
    sums = [0.0] * bits
    for feature, weight in vector.items():
        base = feature.encode("utf-8")
        for j in range(bits):
            h = hashlib.blake2b(base + j.to_bytes(2, "little"),
                                digest_size=8).digest()
            sign = 1.0 if h[0] & 1 else -1.0
            sums[j] += sign * weight
    out = bytearray((bits + 7) // 8)
    for j, s in enumerate(sums):
        if s >= 0:
            out[j >> 3] |= 1 << (j & 7)
    return bytes(out)


# ---------------------------------------------------------------------------
# 稀疏向量工具
# ---------------------------------------------------------------------------

def sparse_dot(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(f, 0.0) for f, w in a.items())


def sparse_norm(v: dict[str, float]) -> float:
    return math.sqrt(sum(w * w for w in v.values()))


def normalize_sparse(v: dict[str, float]) -> tuple[dict[str, float], float]:
    n = sparse_norm(v)
    if n == 0:
        return {}, 0.0
    return {f: w / n for f, w in v.items()}, n


# ---------------------------------------------------------------------------
# 相似度模型：IDF + PPMI 二阶上下文扩展
# ---------------------------------------------------------------------------

class SimilarityModel:
    """全库统计量：词 IDF 与 PPMI 上下文扩展矩阵。

    - :meth:`idf` 支撑词法 TF-IDF；
    - :attr:`expansion` 是 ``词 -> {上下文词: 归一化PPMI权重}``，
      文档语义向量据此做二阶聚合（同义词因上下文分布相似而靠近）。
    """

    def __init__(self, config: Optional[dict] = None,
                 segmenter: Optional[Segmenter] = None):
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self.segmenter = segmenter or _DEFAULT_SEG
        self.n_docs = 0
        self.idf: dict[str, float] = {}
        self.char_idf: dict[str, float] = {}
        self.expansion: dict[str, dict[str, float]] = {}

    # -- 训练 -------------------------------------------------------------
    def fit(self, documents: list[tuple[str, list[str]]]) -> "SimilarityModel":
        """从 ``(原文, 内容词)`` 列表统计 IDF 与共现矩阵。"""
        cfg = self.config
        self.n_docs = len(documents)

        df: Counter = Counter()
        cdf: Counter = Counter()
        tf_all: Counter = Counter()
        token_lists: list[list[str]] = []
        for text, tokens in documents:
            token_lists.append(tokens)
            tf_all.update(tokens)
            df.update(set(tokens))
            cdf.update(set(_cjk_chars(text)))

        # IDF（与 nlp.text.compute_tfidf 同一形式）
        n = self.n_docs
        self.idf = {
            w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()
        }
        self.char_idf = {
            ch: math.log((n + 1) / (c + 1)) + 1.0 for ch, c in cdf.items()
        }

        # 词表按全库词频截断
        vocab = {w for w, _ in tf_all.most_common(cfg["vocab_size"])}

        # 对称共现（滑动窗口，距离倒数衰减）
        cooc: dict[str, Counter] = defaultdict(Counter)
        window = cfg["window"]
        word_total: Counter = Counter()
        for tokens in token_lists:
            ids = [i for i, w in enumerate(tokens) if w in vocab]
            pos_set = set(ids)
            for i in ids:
                w = tokens[i]
                word_total[w] += 1
                for j in range(i + 1, min(i + window, len(tokens))):
                    if j not in pos_set:
                        continue
                    c = tokens[j]
                    decay = 1.0 / (j - i)
                    cooc[w][c] += decay
                    cooc[c][w] += decay

        total_ctx = sum(word_total.values()) or 1
        # PPMI + 每行 top-k + 行归一化，并加自环（保留一阶锚点）
        self.expansion = {}
        ctx_sum = {w: sum(row.values()) for w, row in cooc.items()}
        for w in vocab:
            row = cooc.get(w)
            feats: dict[str, float] = {}
            if row:
                pw = ctx_sum.get(w, 0.0) / total_ctx
                scored = []
                for c, val in row.items():
                    pc = ctx_sum.get(c, 0.0) / total_ctx
                    if pc <= 0:
                        continue
                    pmi = math.log((val / total_ctx) / (pw * pc) + 1e-12)
                    if pmi > 0:
                        scored.append((c, pmi))
                scored.sort(key=lambda x: (-x[1], x[0]))
                for c, pmi in scored[:cfg["context_topk"]]:
                    feats[c] = pmi
            feats[w] = feats.get(w, 0.0) + 1.0  # 自环锚点
            norm = math.sqrt(sum(x * x for x in feats.values())) or 1.0
            self.expansion[w] = {c: x / norm for c, x in feats.items()}
        return self

    # -- 文档向量化 --------------------------------------------------------
    def _weighted_vec(self, tf: Counter, idf: dict[str, float]) -> dict[str, float]:
        """次线性 TF × IDF，L2 归一化。未见特征按最罕见 IDF 处理。"""
        default_idf = math.log((self.n_docs + 1) / 1) + 1.0
        vec: dict[str, float] = {}
        for w, c in tf.items():
            weight = idf.get(w, default_idf)
            vec[w] = (1.0 + math.log(c)) * weight
        vec, _ = normalize_sparse(vec)
        return vec

    def lexical_vector(self, tf: Counter) -> dict[str, float]:
        return self._weighted_vec(tf, self.idf)

    def char_vector(self, text: str) -> dict[str, float]:
        return self._weighted_vec(char_unigrams(text), self.char_idf)

    def semantic_vector(self, tf: Counter) -> dict[str, float]:
        """词的二阶 PPMI 上下文按词法权重加权求和，L2 归一化。"""
        vec: dict[str, float] = {}
        for w, c in tf.items():
            row = self.expansion.get(w)
            if not row:
                # OOV：把自身作为上下文（与自环同口径）
                vec[w] = vec.get(w, 0.0) + (1.0 + math.log(c))
                continue
            base = (1.0 + math.log(c)) * (self.idf.get(w, 1.0))
            for ctx, ew in row.items():
                vec[ctx] = vec.get(ctx, 0.0) + base * ew
        vec, _ = normalize_sparse(vec)
        return vec

    # -- 序列化（模型体量小，直接 JSON 友好的 dict/list） ------------------
    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "n_docs": self.n_docs,
            "idf": self.idf,
            "char_idf": self.char_idf,
            "expansion": {w: [[c, x] for c, x in row.items()]
                          for w, row in self.expansion.items()},
        }

    @classmethod
    def from_dict(cls, data: dict, segmenter: Optional[Segmenter] = None) -> "SimilarityModel":
        model = cls(data.get("config"), segmenter)
        model.n_docs = data.get("n_docs", 0)
        model.idf = dict(data.get("idf", {}))
        model.char_idf = dict(data.get("char_idf", {}))
        model.expansion = {
            w: {c: x for c, x in row}
            for w, row in data.get("expansion", {}).items()
        }
        return model


# ---------------------------------------------------------------------------
# 文档指纹
# ---------------------------------------------------------------------------

def make_fingerprint(doc_id: Optional[str], text: str,
                     model: SimilarityModel,
                     minhasher: MinHasher,
                     name: str = "") -> dict:
    """为单篇文档生成完整指纹。"""
    tokens = content_tokens(text, model.segmenter)
    tf = Counter(tokens)
    bigrams = char_bigrams(text)
    lex_vec = model.lexical_vector(tf)
    char_vec = model.char_vector(text)
    sem_vec = model.semantic_vector(tf)
    cfg = model.config
    return {
        "id": doc_id,
        "name": name,
        "length": len(text or ""),
        "tokens": len(tokens),
        "tf": dict(tf),
        "bigrams": sorted(bigrams),
        "lex": lex_vec,
        "char": char_vec,
        "sem": sem_vec,
        "sig": minhasher.signature(shingle_set(tokens, bigrams)),
        "sem_sig": list(hyperplane_signature(
            sem_vec, cfg["sem_bits"], cfg["seed"])),
    }


# ---------------------------------------------------------------------------
# 打分
# ---------------------------------------------------------------------------

def cosine_sparse(a: dict[str, float], b: dict[str, float],
                  na: Optional[float] = None, nb: Optional[float] = None) -> float:
    """指纹里的向量已归一化，正常情况下就是点积；保留范数兜底。"""
    dot = sparse_dot(a, b)
    if na is None:
        na = sparse_norm(a) or 1.0
    if nb is None:
        nb = sparse_norm(b) or 1.0
    return max(0.0, min(1.0, dot / (na * nb)))


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def pair_score(fp_a: dict, fp_b: dict, config: dict,
               word_cos: Optional[float] = None) -> dict:
    """计算两篇文档的分项相似度（全部落在 [0, 1]）。"""
    if word_cos is None:
        word_cos = cosine_sparse(fp_a["lex"], fp_b["lex"])
    char_cos = cosine_sparse(fp_a.get("char", {}), fp_b.get("char", {}))
    bg_a = set(fp_a["bigrams"])
    bg_b = set(fp_b["bigrams"])
    bg_jac = jaccard(bg_a, bg_b)
    sem_cos = cosine_sparse(fp_a["sem"], fp_b["sem"])
    lexical = 0.45 * word_cos + 0.30 * char_cos + 0.25 * bg_jac
    sem_weight = config.get("sem_weight", DEFAULT_CONFIG["sem_weight"])
    total = sem_weight * sem_cos + (1.0 - sem_weight) * lexical
    return {
        "score": round(total, 6),
        "lexical": round(lexical, 6),
        "semantic": round(sem_cos, 6),
        "word": round(word_cos, 6),
        "char": round(char_cos, 6),
        "bigram": round(bg_jac, 6),
    }


# ---------------------------------------------------------------------------
# LSH 候选对
# ---------------------------------------------------------------------------

def _bits_set(value: int) -> int:
    return bin(value).count("1")


def lsh_candidate_pairs(fps: list[dict], config: dict) -> set[tuple[str, str]]:
    """词法 MinHash-LSH 与语义超平面 LSH 的候选对并集。"""
    pairs: set[tuple[str, str]] = set()
    if len(fps) < 2:
        return pairs

    def _emit(buckets: dict):
        for members in buckets.values():
            if len(members) < 2:
                continue
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    a, b = members[x], members[y]
                    pairs.add((a, b) if a < b else (b, a))

    # 词法：每个 band 以该 band 的签名切片为桶键
    bands, rows = config["lex_bands"], config["lex_rows"]
    buckets: dict[tuple[int, tuple], list[str]] = defaultdict(list)
    for fp in fps:
        sig = fp["sig"]
        for b in range(bands):
            key = (b, tuple(sig[b * rows:(b + 1) * rows]))
            buckets[key].append(fp["id"])
    _emit(buckets)

    # 语义：sem_sig 是字节列表（每位一个超平面）
    sb, sr = config["sem_bands"], config["sem_rows"]
    sem_buckets: dict[tuple[int, bytes], list[str]] = defaultdict(list)
    for fp in fps:
        raw = bytes(fp["sem_sig"])
        need = sb * sr
        for b in range(sb):
            chunk = bytearray(sr)
            for r in range(sr):
                pos = b * sr + r
                if pos < need and (raw[pos >> 3] >> (pos & 7)) & 1:
                    chunk[r] = 1
            sem_buckets[(b, bytes(chunk))].append(fp["id"])
    _emit(sem_buckets)
    return pairs


def query_candidates(query_fp: dict, fps: list[dict],
                     config: dict) -> set[str]:
    """单条查询（可能不在库中）通过 LSH 桶召回候选 id。"""
    if len(fps) <= BRUTE_FORCE_LIMIT:
        return {fp["id"] for fp in fps}

    ids: set[str] = set()
    by_id = {fp["id"]: fp for fp in fps}

    def _collect_index_pairs():
        index_pairs = lsh_candidate_pairs(fps, config)
        related = set()
        for a, b in index_pairs:
            if a == query_fp["id"]:
                related.add(b)
            elif b == query_fp["id"]:
                related.add(a)
        return related

    # 查询本身在库中时，直接复用全库桶里它所在的桶
    if query_fp.get("id") and query_fp["id"] in by_id:
        ids |= _collect_index_pairs()

    # 再以查询签名为桶键，与库内指纹逐 band 匹配
    bands, rows = config["lex_bands"], config["lex_rows"]
    q_sig = query_fp["sig"]
    for fp in fps:
        sig = fp["sig"]
        for b in range(bands):
            if tuple(q_sig[b * rows:(b + 1) * rows]) == tuple(sig[b * rows:(b + 1) * rows]):
                ids.add(fp["id"])
                break

    sb, sr = config["sem_bands"], config["sem_rows"]
    q_raw = bytes(query_fp["sem_sig"])
    for fp in fps:
        raw = bytes(fp["sem_sig"])
        for b in range(sb):
            same = True
            for r in range(sr):
                pos = b * sr + r
                q_bit = (q_raw[pos >> 3] >> (pos & 7)) & 1
                f_bit = (raw[pos >> 3] >> (pos & 7)) & 1
                if q_bit != f_bit:
                    same = False
                    break
            if same:
                ids.add(fp["id"])
                break
    return ids


# ---------------------------------------------------------------------------
# 并查集与全库聚类
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        # 以 id 较小者为根，保证组件代表稳定
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def find_duplicate_edges(fps: list[dict], config: dict,
                         dup_threshold: float, lex_floor: float) -> dict:
    """返回 ``{(id_a, id_b): 分项分数}`` 的重复边集合。

    小库全对比较；大库先过双通道 LSH 候选，再用精确打分确认。
    """
    edges: dict[tuple[str, str], dict] = {}
    if len(fps) < 2:
        return edges

    if len(fps) <= BRUTE_FORCE_LIMIT:
        pairs = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                pairs.append((fps[i]["id"], fps[j]["id"]))
    else:
        pairs = sorted(lsh_candidate_pairs(fps, config))

    by_id = {fp["id"]: fp for fp in fps}
    # bigram 集合缓存独立于指纹对象，避免污染持久化数据
    bg_sets = {fp["id"]: set(fp["bigrams"]) for fp in fps}
    sem_weight = config.get("sem_weight", DEFAULT_CONFIG["sem_weight"])
    for a, b in pairs:
        fp_a, fp_b = by_id[a], by_id[b]
        word_cos = cosine_sparse(fp_a["lex"], fp_b["lex"])
        char_cos = cosine_sparse(fp_a.get("char", {}), fp_b.get("char", {}))
        sem_cos = cosine_sparse(fp_a["sem"], fp_b["sem"])
        pre_lex = 0.45 * word_cos + 0.30 * char_cos
        # bigram 上界 = pre_lex + 0.25；总分上界 = (1-w)*lex_upper + w*sem。
        # 上界都过不了阈值就不必算集合交并。
        lex_upper = pre_lex + 0.25
        score_upper = (1.0 - sem_weight) * lex_upper + sem_weight * sem_cos
        if score_upper < dup_threshold or lex_upper < lex_floor:
            continue
        bg_jac = jaccard(bg_sets[a], bg_sets[b])
        lexical = pre_lex + 0.25 * bg_jac
        total = (1.0 - sem_weight) * lexical + sem_weight * sem_cos
        if total >= dup_threshold and lexical >= lex_floor:
            edges[(a, b)] = {
                "score": round(total, 6),
                "lexical": round(lexical, 6),
                "semantic": round(sem_cos, 6),
                "word": round(word_cos, 6),
                "char": round(char_cos, 6),
                "bigram": round(bg_jac, 6),
            }
    return edges


def cluster_edges(fps: list[dict], edges: dict) -> list[dict]:
    """把重复边用并查集归组，返回按组规模/代表 id 稳定排序的分组。"""
    uf = UnionFind(fp["id"] for fp in fps)
    for a, b in edges:
        uf.union(a, b)

    comps: dict[str, list[str]] = defaultdict(list)
    for fp in fps:
        comps[uf.find(fp["id"])].append(fp["id"])

    groups = []
    for _, members in comps.items():
        if len(members) < 2:
            continue
        members.sort()
        # 组内平均相似度最高的文档为代表
        totals: dict[str, float] = defaultdict(float)
        counts: Counter = Counter()
        edge_scores = []
        for (a, b), parts in edges.items():
            if a in members and b in members:
                totals[a] += parts["score"]
                totals[b] += parts["score"]
                counts[a] += 1
                counts[b] += 1
                edge_scores.append(parts["score"])
        representative = min(
            members,
            key=lambda m: (-(totals[m] / counts[m]) if counts[m] else 0.0, m),
        )
        groups.append({
            "docs": members,
            "representative": representative,
            "size": len(members),
            "max_score": round(max(edge_scores), 6) if edge_scores else 0.0,
            "avg_score": round(sum(edge_scores) / len(edge_scores), 6)
            if edge_scores else 0.0,
            "edge_count": len(edge_scores),
        })
    groups.sort(key=lambda g: (-g["size"], -g["max_score"], g["representative"]))
    return groups
