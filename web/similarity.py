"""相似检索与全库查重的服务层。

把 :mod:`nlp.similarity` 的纯算法接到分片存储上，解决三件工程问题：

1. **跨分片汇总排序**：为每篇语料在 ``simindex`` 任务库中持久化一份
   相似度特征（词集合 / 字 shingle / MinHash 签名），检索时一次性
   加载（带文件 mtime 缓存），对全部文档打分后全局排序、再分页——
   排序键为 ``(-score, id)`` 的全序，翻页连续、不重不漏。
2. **增量维护**：语料新增/删除时同步维护索引与既有归组；
   索引与语料出现不一致（如绕过 API 直接改库）时，
   :meth:`SimilarityService.ensure_index` 自动修复（自愈）。
3. **归组稳定**：查重归组结果持久化在 ``dupgroup`` 任务库，
   重新跑查重时按「成员最大重叠」继承旧组 id——
   新增/删除文档不会冲乱已有归组；相似度口径只依赖文档对本身，
   前后一致。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Optional

from nlp import get_segmenter
from nlp.similarity import (ALL_PAIRS_LIMIT, DEFAULT_THRESHOLD, DocFeatures,
                            all_pairs, candidate_pairs, cluster,
                            extract_features, id_sort_key, score_features)
from storage import StoreRegistry, _atomic_write_json, _read_json


class SimilarityService:
    """语料相似度索引、检索与查重归组。"""

    CORPUS_TASK = "corpus"
    INDEX_TASK = "simindex"
    GROUP_TASK = "dupgroup"
    RUN_FILE = "dedup_run.json"

    def __init__(self, registry: StoreRegistry):
        self.registry = registry
        self._lock = threading.RLock()
        self._feat_cache_key: Optional[tuple] = None
        self._feat_cache: dict[str, DocFeatures] = {}
        self._ensure_cache_key: Optional[tuple] = None

    # ------------------------------------------------------------------
    # 存储句柄
    # ------------------------------------------------------------------
    def _corpus_store(self):
        return self.registry.task(self.CORPUS_TASK)

    def _index_store(self):
        return self.registry.task(self.INDEX_TASK)

    def _group_store(self):
        return self.registry.task(self.GROUP_TASK)

    def _run_path(self) -> str:
        return os.path.join(self.registry.root, self.RUN_FILE)

    # ------------------------------------------------------------------
    # 指纹与缓存（保证「库没变就不重算」，变了一定重算）
    # ------------------------------------------------------------------
    @staticmethod
    def _store_fingerprint(store) -> tuple:
        """用分片文件的 (mtime, size) 刻画存储当前状态。"""
        parts = []
        if os.path.isdir(store.dir):
            for name in sorted(os.listdir(store.dir)):
                if not name.endswith(".json"):
                    continue
                path = os.path.join(store.dir, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                parts.append((name, st.st_mtime_ns, st.st_size))
        return tuple(parts)

    @staticmethod
    def _corpus_fingerprint(live_ids) -> str:
        """语料成员指纹：用于判断查重结果是否已过期。"""
        joined = "".join(sorted(live_ids, key=id_sort_key))
        return hashlib.md5(joined.encode("utf-8")).hexdigest()[:12]

    def _live_corpus(self) -> dict:
        return {r["id"]: r for r in self._corpus_store().all()
                if not r.get("_deleted")}

    # ------------------------------------------------------------------
    # 索引维护
    # ------------------------------------------------------------------
    def ensure_index(self) -> dict:
        """让 ``simindex`` 与语料库保持一致（增量、自愈）。

        - 语料有而索引无 → 补建特征；
        - 索引有而语料无（已删除）→ 清理索引记录。
        """
        with self._lock:
            corpus_fp = self._store_fingerprint(self._corpus_store())
            index_fp = self._store_fingerprint(self._index_store())
            key = (corpus_fp, index_fp)
            if self._ensure_cache_key == key:
                return {"added": 0, "removed": 0,
                        "total": len(self._feat_cache)
                        if self._feat_cache_key else -1,
                        "cached": True}

            live = self._live_corpus()
            indexed = {r["id"] for r in self._index_store().all()
                       if not r.get("_deleted")}

            removed = 0
            for rid in sorted(indexed - set(live), key=id_sort_key):
                if self._index_store().delete(rid):
                    removed += 1

            missing = [cid for cid in live if cid not in indexed]
            if missing:
                segmenter = get_segmenter()
                records = []
                for cid in sorted(missing, key=id_sort_key):
                    record = self._index_record(
                        cid, live[cid].get("text", ""), segmenter)
                    records.append(record)
                self._index_store().insert_many(records)

            self._ensure_cache_key = (
                self._store_fingerprint(self._corpus_store()),
                self._store_fingerprint(self._index_store()),
            )
            return {"added": len(missing), "removed": removed,
                    "total": len(live), "cached": False}

    def reindex(self) -> dict:
        """全量重建索引（清除既有索引记录后重算）。"""
        with self._lock:
            for record in self._index_store().all():
                if not record.get("_deleted"):
                    self._index_store().delete(record["id"])
            self._ensure_cache_key = None
            return self.ensure_index()

    @staticmethod
    def _index_record(cid: str, text: str, segmenter) -> dict:
        features = extract_features(text, segmenter)
        record = features.to_record()
        record.update({
            "id": cid,
            "text_len": len(text or ""),
            "indexed_at": time.time(),
        })
        return record

    def on_corpus_added(self, cid: str, text: str) -> None:
        """语料新增钩子：立即为该文档建立索引。"""
        with self._lock:
            self._index_store().insert(
                self._index_record(cid, text, get_segmenter()))
            self._invalidate()

    def on_corpus_deleted(self, cid: str) -> None:
        """语料删除钩子：清理索引，并把该文档移出既有归组。

        只移除该成员本身：组还剩 ≥2 人时更新记录，不足 2 人时解散；
        其它组完全不受影响（不被冲乱）。
        """
        with self._lock:
            self._index_store().delete(cid)
            features = self._index_features()
            for record in self._group_records():
                members = record.get("members") or []
                if cid not in members:
                    continue
                gid = record["gid"]
                remaining = [m for m in members if m != cid]
                self._group_store().delete(record["id"])
                if len(remaining) >= 2:
                    updated = self._build_group_record(
                        gid, remaining, features,
                        record.get("threshold", DEFAULT_THRESHOLD),
                        created_at=record.get("created_at"),
                        center=record.get("representative"))
                    self._group_store().insert(updated)
            self._invalidate()

    def _invalidate(self) -> None:
        self._feat_cache_key = None
        self._ensure_cache_key = None

    # ------------------------------------------------------------------
    # 特征加载（带缓存）
    # ------------------------------------------------------------------
    def _index_features(self) -> dict[str, DocFeatures]:
        store = self._index_store()
        key = self._store_fingerprint(store)
        if self._feat_cache_key == key:
            return self._feat_cache
        features = {}
        for record in store.all():
            if record.get("_deleted"):
                continue
            features[record["id"]] = DocFeatures.from_record(record)
        self._feat_cache_key = key
        self._feat_cache = features
        return features

    # ------------------------------------------------------------------
    # 相似检索：跨分片汇总 + 全局排序 + 稳定分页
    # ------------------------------------------------------------------
    def search(self, text: Optional[str] = None,
               corpus_id: Optional[str] = None,
               limit: int = 10, offset: int = 0,
               min_score: float = 0.0,
               include_self: bool = False) -> dict:
        """对任意文本（或库内文档）检索最相似的语料，按相似度降序。

        排序键 ``(-score, id)`` 是确定性的全序：只要语料库不变，
        同一查询的任意 ``offset/limit`` 切片拼起来恰好是完整结果，
        翻页不重不漏。
        """
        started = time.time()
        with self._lock:
            self.ensure_index()
            features = self._index_features()

            query_id = None
            if text:
                query_features = extract_features(text, get_segmenter())
            elif corpus_id:
                query_id = corpus_id
                query_features = features.get(corpus_id)
                if query_features is None:
                    record = self._corpus_store().get(corpus_id)
                    if not record or record.get("_deleted"):
                        raise ValueError(f"语料不存在: {corpus_id}")
                    query_features = extract_features(
                        record.get("text", ""), get_segmenter())
            else:
                raise ValueError("缺少查询文本或语料 id")

            scored = []
            for cid, doc_features in features.items():
                if query_id and cid == query_id and not include_self:
                    continue
                result = score_features(query_features, doc_features)
                if result["score"] < min_score:
                    continue
                scored.append((cid, result))

            scored.sort(key=lambda item: (-item[1]["score"],
                                          id_sort_key(item[0])))
            total = len(scored)
            page = scored[offset:offset + limit]

            live = self._live_corpus()
            results = []
            for cid, result in page:
                record = live.get(cid) or {}
                text_full = record.get("text", "")
                results.append({
                    "corpus_id": cid,
                    "name": record.get("name", "未命名"),
                    "score": round(result["score"], 4),
                    "components": {
                        key: round(value, 4) for key, value in result.items()
                        if key != "score"
                    },
                    "length": len(text_full),
                    "preview": text_full[:80],
                })
            return {
                "query_corpus_id": query_id,
                "total": total,
                "offset": offset,
                "limit": limit,
                "min_score": min_score,
                "elapsed_ms": round((time.time() - started) * 1000, 1),
                "results": results,
            }

    # ------------------------------------------------------------------
    # 全库查重：候选对 → 精确复核 → 连通分量 → 归组持久化
    # ------------------------------------------------------------------
    def run_dedup(self, threshold: float = DEFAULT_THRESHOLD) -> dict:
        """对整个语料库做一轮查重，把高度相似的文档归成一组。

        流程：LSH 分带找候选对（小库直接全配对）→ 逐对精确打分 →
        ≥ 阈值的边做星形贪心聚类（成员都与组中心相似，避免传递链误判）→
        与既有归组按成员最大重叠调和组 id（已有归组不被冲乱）→
        整体重写 ``dupgroup`` 库。
        """
        started = time.time()
        with self._lock:
            self.ensure_index()
            features = self._index_features()
            ids = sorted(features, key=id_sort_key)

            # 1. 候选对
            if len(ids) <= ALL_PAIRS_LIMIT:
                candidates = all_pairs(ids)
                candidate_mode = "all_pairs"
            else:
                candidates = candidate_pairs(
                    {cid: features[cid].signature for cid in ids})
                candidate_mode = "lsh"

            # 2. 精确复核
            edges = []
            for a, b in sorted(candidates):
                score = score_features(features[a], features[b])["score"]
                if score >= threshold:
                    edges.append((a, b, score))

            # 3. 星形聚类归组（成员都与组中心高度相似，避免传递链误判）
            components = cluster(edges)

            # 4. 组 id 调和：新分量与旧组按成员重叠最大者继承 gid
            old_groups = {rec["gid"]: rec for rec in self._group_records()}
            remaining = dict(old_groups)
            new_records = []
            for center, members in components:
                gid, created_at = self._inherit_gid(members, remaining,
                                                    old_groups)
                record = self._build_group_record(
                    gid, members, features, threshold, created_at,
                    center=center)
                new_records.append(record)

            # 5. 整体重写组库（旧记录逻辑删除，新记录写入）
            group_store = self._group_store()
            for record in group_store.all():
                if not record.get("_deleted"):
                    group_store.delete(record["id"])
            if new_records:
                group_store.insert_many(new_records)

            live = self._live_corpus()
            run_info = {
                "threshold": threshold,
                "ran_at": time.time(),
                "fingerprint": self._corpus_fingerprint(live.keys()),
                "doc_count": len(ids),
                "candidate_mode": candidate_mode,
                "candidate_pairs": len(candidates),
                "similar_pairs": len(edges),
                "groups": len(new_records),
                "docs_grouped": sum(r["size"] for r in new_records),
                "elapsed_ms": round((time.time() - started) * 1000, 1),
            }
            _atomic_write_json(self._run_path(), run_info)
            return dict(run_info, ok=True)

    @staticmethod
    def _inherit_gid(members: list, remaining: dict, old_groups: dict):
        """为新分量选择组 id：与某个旧组有成员重叠时继承其 gid。"""
        member_set = set(members)
        best_gid, best_overlap = None, 0
        for gid, record in remaining.items():
            overlap = len(member_set & set(record.get("members") or []))
            if overlap > best_overlap:
                best_gid, best_overlap = gid, overlap
        if best_gid is not None:
            del remaining[best_gid]
            return best_gid, old_groups[best_gid].get("created_at")
        # 新组：id 取最小组内成员序号，冲突时追加序号，保证确定且唯一
        base = f"grp_{id_sort_key(members[0])[1]:04d}" \
            if id_sort_key(members[0])[0] == 0 else f"grp_{members[0]}"
        gid, suffix = base, 2
        taken = {r["gid"] for r in old_groups.values()}
        while gid in taken:
            gid = f"{base}_{suffix}"
            suffix += 1
        return gid, None

    def _build_group_record(self, gid: str, members: list,
                            features: dict, threshold: float,
                            created_at: Optional[float] = None,
                            center: Optional[str] = None) -> dict:
        """构造组记录：组内两两打分，统计平均分与每人最高相似度。"""
        ordered = sorted(members, key=id_sort_key)
        member_best = {m: 0.0 for m in ordered}
        pair_scores = []
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                if a not in features or b not in features:
                    continue
                score = score_features(features[a], features[b])["score"]
                pair_scores.append(score)
                member_best[a] = max(member_best[a], score)
                member_best[b] = max(member_best[b], score)
        avg = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
        now = time.time()
        return {
            "gid": gid,
            "members": ordered,
            "member_scores": {m: round(member_best[m], 4) for m in ordered},
            "representative": center if center in ordered else ordered[0],
            "size": len(ordered),
            "avg_score": round(avg, 4),
            "threshold": threshold,
            "created_at": created_at or now,
            "updated_at": now,
        }

    # ------------------------------------------------------------------
    # 归组查询
    # ------------------------------------------------------------------
    def _group_records(self) -> list:
        """当前生效的组记录（按 gid 去重，跳过墓碑）。"""
        records = {}
        for record in self._group_store().all():
            if record.get("_deleted"):
                continue
            gid = record.get("gid")
            # 同一 gid 可能因更新留下多条（旧的已被墓碑标记），
            # 这里再按 updated_at 兜底取最新。
            if gid not in records or \
                    record.get("updated_at", 0) > records[gid].get("updated_at", 0):
                records[gid] = record
        return list(records.values())

    def groups(self) -> dict:
        """列出最近一次查重的归组结果，并标注是否已过期。"""
        with self._lock:
            run_info = _read_json(self._run_path(), None)
            live = self._live_corpus()
            current_fp = self._corpus_fingerprint(live.keys())
            records = self._group_records()
            records.sort(key=lambda r: (-r.get("size", 0),
                                        id_sort_key(r.get("representative", ""))))

            groups = []
            for record in records:
                members = []
                for cid in record.get("members") or []:
                    doc = live.get(cid)
                    if doc is None:
                        continue  # 已被删除（正常会被钩子清理，这里兜底）
                    members.append({
                        "corpus_id": cid,
                        "name": doc.get("name", "未命名"),
                        "preview": (doc.get("text") or "")[:60],
                        "best_score": (record.get("member_scores") or {}).get(cid),
                    })
                if len(members) < 2:
                    continue
                groups.append({
                    "gid": record["gid"],
                    "size": len(members),
                    "avg_score": record.get("avg_score"),
                    "representative": record.get("representative"),
                    "created_at": record.get("created_at"),
                    "updated_at": record.get("updated_at"),
                    "members": members,
                })

            stale = bool(run_info) and \
                run_info.get("fingerprint") != current_fp
            return {
                "groups": groups,
                "group_count": len(groups),
                "docs_grouped": sum(g["size"] for g in groups),
                "corpus_size": len(live),
                "threshold": (run_info or {}).get("threshold"),
                "ran_at": (run_info or {}).get("ran_at"),
                "stale": stale,
                "never_run": run_info is None,
            }
