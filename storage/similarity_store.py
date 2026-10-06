"""相似度索引的分片持久化与服务层。

文件布局（与 :class:`~storage.sharded.ShardedStore` 同一风格）::

    data/similarity/
      meta.json            # 索引元数据：口径版本、corpus 修订、分片数
      model.json           # SimilarityModel（IDF + PPMI 扩展矩阵）
      groups.json          # 查重归组（稳定 group_id + 所用阈值）
      fp_shard_000000.json # 指纹分片，每个分片最多 shard_size 条

关键性质
========
* **口径一致**：指纹方案版本、模型参数随 meta.json 固化；检测到代码版本
  升级或语料变更才重建，重建后 ``index_version`` 单调递增。
* **增量友好**：新增/删除语料触发一次全量重建（模型是全库统计量），
  但归组身份通过「按成员重叠度继承旧 group_id」保持稳定：新簇优先继承
  重叠最多的旧组，分裂时最大的碎片保留原 id，合并时保留最老的 id；
  无法继承的簇才分配新的自增 id。
* **分页连续**：检索排序键为 ``(score desc, doc_id asc)``，游标编码
  末条的 (index_version, score, id)，翻页按严格序比较，不重不漏；
  语料变更使 index_version 失配时响应里明确告知 ``stale_cursor``。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from typing import Optional

from nlp.similarity import (DEFAULT_CONFIG, MinHasher, SimilarityModel,
                            cluster_edges, content_tokens, find_duplicate_edges,
                            make_fingerprint, pair_score, query_candidates)
from .lock import FileLock, lock_path_for
from .sharded import _atomic_write_json, _read_json

# 查重默认阈值（阈值不影响指纹，调整无需重建索引）
DEFAULT_DUP_THRESHOLD = 0.72
DEFAULT_LEX_FLOOR = 0.35

FP_SHARD_SIZE = 256


def _encode_cursor(index_version: int, score: float, doc_id: str) -> str:
    raw = json.dumps({"v": index_version, "s": score, "id": doc_id},
                     ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_cursor(cursor: str) -> Optional[dict]:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            return None
        if "v" in data and "s" in data and "id" in data:
            return data
    except (ValueError, UnicodeDecodeError):
        return None
    return None


class SimilarityService:
    """语料库相似度检索 / 查重服务。"""

    def __init__(self, registry, task: str = "corpus",
                 dirname: str = "similarity",
                 shard_size: int = FP_SHARD_SIZE):
        self.registry = registry
        self.task = task
        self.dir = os.path.join(registry.root, dirname)
        self.meta_path = os.path.join(self.dir, "meta.json")
        self.model_path = os.path.join(self.dir, "model.json")
        self.groups_path = os.path.join(self.dir, "groups.json")
        self.shard_size = shard_size
        os.makedirs(self.dir, exist_ok=True)

        self._mem_lock = threading.RLock()
        self._rebuild_lock = threading.Lock()
        self._state: Optional[dict] = None  # 内存索引缓存

    # -- 路径与底层读写 ----------------------------------------------------
    def _shard_path(self, index: int) -> str:
        return os.path.join(self.dir, f"fp_shard_{index:06d}.json")

    def _read_meta(self) -> Optional[dict]:
        return _read_json(self.meta_path, None)

    def _read_groups(self) -> Optional[dict]:
        return _read_json(self.groups_path, None)

    def _write_groups(self, data: dict) -> None:
        _atomic_write_json(self.groups_path, data)

    def _read_shard(self, index: int) -> list:
        data = _read_json(self._shard_path(index), [])
        return data if isinstance(data, list) else []

    def _corpus_snapshot(self) -> tuple[list[dict], tuple]:
        """读取语料库存活文档，返回 (文档列表, 廉价变更签名)。"""
        store = self.registry.task(self.task)
        records = [r for r in store.all() if not r.get("_deleted")]
        records.sort(key=lambda r: r.get("id", ""))
        stats = store.stats()
        sig = (stats.get("total", 0), stats.get("shard_count", 0))
        return records, sig

    @staticmethod
    def _content_revision(records: list[dict]) -> str:
        h = hashlib.sha256()
        for r in records:
            text = r.get("text", "")
            h.update(r.get("id", "").encode("utf-8"))
            h.update(b"\0")
            h.update(hashlib.sha1(text.encode("utf-8")).digest())
            h.update(b"\n")
        return h.hexdigest()

    # -- 索引加载 / 同步 ---------------------------------------------------
    def _load_state(self) -> Optional[dict]:
        meta = self._read_meta()
        if not meta:
            return None
        model_data = _read_json(self.model_path, None)
        if model_data is None:
            return None
        fps = []
        for index in range(meta.get("shard_count", 0)):
            fps.extend(self._read_shard(index))
        model = SimilarityModel.from_dict(model_data)
        return {
            "meta": meta,
            "model": model,
            "fps": fps,
            "by_id": {fp["id"]: fp for fp in fps},
            "minhasher": MinHasher(
                model.config["minhash_perm"], model.config["seed"]),
            "corpus_sig": tuple(meta.get("corpus_sig", [])),
        }

    def _get_state(self) -> Optional[dict]:
        with self._mem_lock:
            if self._state is not None:
                return self._state
            # 读路径用共享锁，避免读到重建写一半的元数据
            with FileLock(lock_path_for(self.meta_path), mode="shared"):
                self._state = self._load_state()
            return self._state

    def ensure_index(self, force: bool = False) -> dict:
        """确保索引与语料库一致；必要时重建。返回状态摘要。"""
        records, sig = self._corpus_snapshot()
        state = self._get_state()
        # 廉价签名 (存活数, 分片数) 只在增删/compact 时变化，语料没有
        # 原地编辑接口，因此签名一致即内容一致，无需每请求全文哈希。
        up_to_date = (
            not force and state is not None
            and state["meta"].get("fp_version") == DEFAULT_CONFIG["fp_version"]
            and state["corpus_sig"] == sig
        )
        if up_to_date:
            return self.status()

        with self._rebuild_lock:
            # 双检：可能已被别的线程重建
            state = self._get_state()
            if not force and state is not None and \
                    state["meta"].get("fp_version") == DEFAULT_CONFIG["fp_version"] and \
                    state["corpus_sig"] == sig:
                return self.status()
            rev = self._content_revision(records)
            self._rebuild(records, sig, rev)
        return self.status()

    def _rebuild(self, records: list[dict], sig: tuple, rev: str) -> None:
        old_meta = self._read_meta() or {}
        prev_version = old_meta.get("index_version", 0)
        score_cfg = old_meta.get("score_config", {})

        model = SimilarityModel(DEFAULT_CONFIG)
        token_lists = [content_tokens(r.get("text", ""), model.segmenter)
                       for r in records]
        model.fit([(r.get("text", ""), tokens)
                   for r, tokens in zip(records, token_lists)])
        minhasher = MinHasher(model.config["minhash_perm"],
                              model.config["seed"])

        fps = []
        for record, tokens in zip(records, token_lists):
            text = record.get("text", "")
            fp = make_fingerprint(
                record.get("id"), text, model, minhasher,
                name=record.get("name", ""))
            fp["preview"] = text[:80]
            fps.append(fp)
        fps.sort(key=lambda fp: fp["id"])

        groups_data = self._refresh_groups(fps, model.config, old_meta,
                                           rebuild=True)

        with FileLock(lock_path_for(self.meta_path)):
            # 清掉旧分片（数量可能变化），再写新分片
            existing = sorted(f for f in os.listdir(self.dir)
                              if f.startswith("fp_shard_"))
            for name in existing:
                try:
                    os.remove(os.path.join(self.dir, name))
                except FileNotFoundError:
                    pass
            shard_count = 0
            for start in range(0, len(fps), self.shard_size):
                chunk = fps[start:start + self.shard_size]
                _atomic_write_json(self._shard_path(shard_count), chunk)
                shard_count += 1

            _atomic_write_json(self.model_path, model.to_dict())
            if groups_data is not None:
                self._write_groups(groups_data)

            meta = {
                "fp_version": DEFAULT_CONFIG["fp_version"],
                "fingerprint_config": model.config,
                "score_config": {
                    "sem_weight": score_cfg.get(
                        "sem_weight", DEFAULT_CONFIG["sem_weight"]),
                },
                "corpus_rev": rev,
                "corpus_sig": list(sig),
                "index_version": prev_version + 1,
                "shard_count": shard_count,
                "shard_size": self.shard_size,
                "doc_count": len(fps),
                "built_at": time.time(),
            }
            _atomic_write_json(self.meta_path, meta)

        with self._mem_lock:
            self._state = None  # 下次读取时加载新索引

    # -- 归组：稳定继承 ----------------------------------------------------
    @staticmethod
    def _inherit_group_ids(new_groups: list[dict],
                           old_groups: list[dict]) -> dict[str, str]:
        """为每个新簇挑选可继承的旧 group_id。

        返回 ``{新簇代表 doc_id: group_id}``。贪心处理：簇规模大的先选，
        每个旧组最多被继承一次，按「重叠数 -> Jaccard -> 更老的组」排序。
        """
        old_sets = {g["group_id"]: set(g["docs"]) for g in old_groups}
        used: set[str] = set()
        mapping: dict[str, str] = {}
        ordered = sorted(new_groups,
                         key=lambda g: (-g["size"], g["representative"]))
        for group in ordered:
            members = set(group["docs"])
            best_id, best_key = None, None
            for gid, old_members in old_sets.items():
                if gid in used:
                    continue
                overlap = len(members & old_members)
                if overlap == 0:
                    continue
                union = len(members | old_members)
                key = (overlap, overlap / union,
                       -_group_seq(gid))
                if best_key is None or key > best_key:
                    best_key, best_id = key, gid
            if best_id:
                used.add(best_id)
                mapping[group["representative"]] = best_id
        return mapping

    def _refresh_groups(self, fps, fp_config, old_meta,
                        rebuild: bool,
                        dup_threshold: float = DEFAULT_DUP_THRESHOLD,
                        lex_floor: float = DEFAULT_LEX_FLOOR) -> Optional[dict]:
        """重算查重边并归组。

        重建（语料变更）时仅在历史上跑过查重的情况下自动刷新，
        并沿用旧阈值；显式查重可指定新阈值。
        """
        old_data = self._read_groups()
        if rebuild and old_data is None:
            return None
        if old_data is None:
            old_data = {"group_seq": 0, "groups": []}
        if rebuild:
            dup_threshold = old_data.get("dup_threshold", dup_threshold)
            lex_floor = old_data.get("lex_floor", lex_floor)

        effective = dict(fp_config)
        effective["sem_weight"] = old_meta.get(
            "score_config", {}).get(
            "sem_weight", DEFAULT_CONFIG["sem_weight"])

        edges = find_duplicate_edges(fps, effective, dup_threshold, lex_floor)
        groups = cluster_edges(fps, edges)
        mapping = self._inherit_group_ids(groups, old_data.get("groups", []))

        seq = old_data.get("group_seq", 0)
        enriched = []
        for group in groups:
            gid = mapping.get(group["representative"])
            if gid is None:
                seq += 1
                gid = f"dup_{seq:04d}"
            item = dict(group)
            item["group_id"] = gid
            enriched.append(item)
        enriched.sort(key=lambda g: g["group_id"])

        return {
            "group_seq": seq,
            "dup_threshold": dup_threshold,
            "lex_floor": lex_floor,
            "updated_at": time.time(),
            "groups": enriched,
        }

    # -- 检索 --------------------------------------------------------------
    def _effective_config(self, state: dict,
                          sem_weight: Optional[float]) -> dict:
        cfg = dict(state["model"].config)
        if sem_weight is None:
            sem_weight = state["meta"].get(
                "score_config", {}).get(
                "sem_weight", DEFAULT_CONFIG["sem_weight"])
        cfg["sem_weight"] = sem_weight
        return cfg

    def search(self, text: Optional[str] = None, doc_id: Optional[str] = None,
               limit: int = 10, cursor: Optional[str] = None,
               sem_weight: Optional[float] = None) -> dict:
        """跨全部分片检索相似文档，按 (score desc, id asc) 游标分页。"""
        self.ensure_index()
        loaded = self._get_state()
        if loaded is None or not loaded["fps"]:
            return {"results": [], "total": 0, "next_cursor": None,
                    "index_version": (loaded["meta"]["index_version"]
                                      if loaded else 0),
                    "stale_cursor": False,
                    "error": "索引为空，请先上传语料"}

        index_version = loaded["meta"]["index_version"]
        cfg = self._effective_config(loaded, sem_weight)

        # 查询指纹：库内文档直接复用，否则现算
        if doc_id:
            query_fp = loaded["by_id"].get(doc_id)
            if query_fp is None:
                return {"results": [], "total": 0, "next_cursor": None,
                        "index_version": index_version, "stale_cursor": False,
                        "error": f"语料不存在: {doc_id}"}
            exclude = {doc_id}
        else:
            if not text or not text.strip():
                return {"results": [], "total": 0, "next_cursor": None,
                        "index_version": index_version, "stale_cursor": False,
                        "error": "缺少查询文本"}
            query_fp = make_fingerprint(None, text, loaded["model"],
                                        loaded["minhasher"])
            exclude = set()

        cursor_info = _decode_cursor(cursor) if cursor else None
        stale_cursor = bool(cursor_info and
                            cursor_info.get("v") != index_version)

        candidate_ids = query_candidates(query_fp, loaded["fps"], cfg)
        scored = []
        for cid in candidate_ids:
            if cid in exclude:
                continue
            parts = pair_score(query_fp, loaded["by_id"][cid], cfg)
            scored.append((cid, parts))
        scored.sort(key=lambda x: (-x[1]["score"], x[0]))

        # 游标：严格序 (score desc, id asc) 中排在 (s, id) 之后的项
        page = []
        if cursor_info and not stale_cursor:
            cs, cid = cursor_info["s"], cursor_info["id"]
            filtered = []
            for rid, parts in scored:
                if parts["score"] < cs or \
                        (parts["score"] == cs and rid > cid):
                    filtered.append((rid, parts))
            scored = filtered

        page = scored[:limit]
        next_cursor = None
        if len(scored) > limit:
            last_id, last_parts = page[-1]
            next_cursor = _encode_cursor(
                index_version, last_parts["score"], last_id)

        results = []
        for rid, parts in page:
            fp = loaded["by_id"][rid]
            results.append({
                "id": rid,
                "name": fp.get("name", ""),
                "preview": fp.get("preview", ""),
                "length": fp.get("length", 0),
                **parts,
            })
        return {
            "results": results,
            "total_returned": len(results),
            "candidate_count": len(candidate_ids) - len(exclude),
            "next_cursor": next_cursor,
            "index_version": index_version,
            "stale_cursor": stale_cursor,
            "thresholds": {
                "sem_weight": cfg["sem_weight"],
            },
        }

    # -- 全库查重 ----------------------------------------------------------
    def run_dedup(self, dup_threshold: float = DEFAULT_DUP_THRESHOLD,
                  lex_floor: float = DEFAULT_LEX_FLOOR,
                  force_rebuild: bool = False) -> dict:
        self.ensure_index(force=force_rebuild)
        loaded = self._get_state()
        if loaded is None:
            return {"groups": [], "group_count": 0, "doc_count": 0,
                    "index_version": 0, "edge_count": 0}

        meta = loaded["meta"]
        groups_data = self._refresh_groups(
            loaded["fps"], loaded["model"].config, meta, rebuild=False,
            dup_threshold=dup_threshold, lex_floor=lex_floor)
        groups_data["index_version"] = meta["index_version"]
        with FileLock(lock_path_for(self.groups_path)):
            self._write_groups(groups_data)

        by_id = loaded["by_id"]
        groups = []
        edge_total = 0
        for g in groups_data["groups"]:
            edge_total += g["edge_count"]
            groups.append({
                **g,
                "members": [{
                    "id": d,
                    "name": by_id[d].get("name", ""),
                    "preview": by_id[d].get("preview", ""),
                    "length": by_id[d].get("length", 0),
                } for d in g["docs"]],
            })
        return {
            "groups": groups,
            "group_count": len(groups),
            "edge_count": edge_total,
            "doc_count": len(loaded["fps"]),
            "index_version": meta["index_version"],
            "dup_threshold": dup_threshold,
            "lex_floor": lex_floor,
            "elapsed_note": "小库全对比较；大库走双通道 LSH 候选 + 精排",
        }

    def get_groups(self) -> Optional[dict]:
        loaded = self._get_state()
        data = self._read_groups()
        if data is None:
            return None
        by_id = loaded["by_id"] if loaded else {}
        groups = []
        for g in data.get("groups", []):
            # 仅返回仍在库中的成员（删除后未重建的兜底）
            live_members = [d for d in g["docs"] if d in by_id]
            if len(live_members) < 2:
                continue
            groups.append({
                **g,
                "docs": live_members,
                "size": len(live_members),
                "members": [{
                    "id": d,
                    "name": by_id[d].get("name", ""),
                    "preview": by_id[d].get("preview", ""),
                    "length": by_id[d].get("length", 0),
                } for d in live_members],
            })
        return {
            "groups": groups,
            "group_count": len(groups),
            "dup_threshold": data.get("dup_threshold", DEFAULT_DUP_THRESHOLD),
            "lex_floor": data.get("lex_floor", DEFAULT_LEX_FLOOR),
            "index_version": data.get("index_version"),
            "updated_at": data.get("updated_at"),
        }

    # -- 状态 --------------------------------------------------------------
    def status(self) -> dict:
        loaded = self._get_state()
        records, sig = self._corpus_snapshot()
        if loaded is None:
            return {
                "built": False,
                "corpus_count": len(records),
                "doc_count": 0,
            }
        meta = loaded["meta"]
        groups_data = self._read_groups()
        return {
            "built": True,
            "fp_version": meta.get("fp_version"),
            "index_version": meta.get("index_version"),
            "doc_count": meta.get("doc_count", 0),
            "corpus_count": len(records),
            "corpus_sig": list(sig),
            "in_sync": (meta.get("corpus_rev") == self._content_revision(records)
                        and meta.get("corpus_sig") == list(sig)),
            "shard_count": meta.get("shard_count", 0),
            "built_at": meta.get("built_at"),
            "score_config": meta.get("score_config", {}),
            "fingerprint_config": meta.get("fingerprint_config", {}),
            "dedup_ran": groups_data is not None,
            "group_count": len(groups_data.get("groups", [])) if groups_data else 0,
        }


def _group_seq(group_id: str) -> int:
    try:
        return int(group_id.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0
