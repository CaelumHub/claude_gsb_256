"""分片 JSON 存储引擎。

这是本项目「多任务结果存储 / JSON 分片合并与查询」难点的核心实现。

设计目标
--------
1. **按任务分片**：每种任务（seg / pos / ner / sentiment / ...）拥有独立目录，
   目录内是若干个大小受控的分片文件 ``shard_000001.json``。
2. **并发安全**：所有读-改-写都通过 :class:`~storage.lock.FileLock` 串行化，
   写文件采用「临时文件 + 原子替换」，避免进程崩溃留下损坏文件。
3. **可查询**：提供轻量查询语言（字段过滤 + 排序 + 分页），跨分片扫描并合并。
4. **可合并**：提供 ``compact`` / ``merge`` 把多个分片归并成一个有序整体，
   同时清理删除标记（墓碑）。

文件布局
--------
    data/
      <task>/
        meta.json            # 元数据：分片数、记录总数、分片大小、自增 id
        shard_000001.json    # 记录数组，最多 shard_size 条
        shard_000002.json
        ...
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Callable, Iterable, Iterator, Optional

from .lock import FileLock, lock_path_for


# 默认每个分片最多容纳的记录数
DEFAULT_SHARD_SIZE = 100

# 查询支持的比较操作符
_OPERATORS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
    "contains": lambda a, b: b in a,
    "startswith": lambda a, b: a.startswith(b),
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
}


def _atomic_write_json(path: str, obj: Any) -> None:
    """原子写入 JSON：先写临时文件，再 ``os.replace`` 覆盖。"""
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class ShardedStore:
    """单个任务的 JSON 分片存储。"""

    def __init__(self, root: str, task: str, shard_size: int = DEFAULT_SHARD_SIZE):
        self.task = task
        self.shard_size = max(1, int(shard_size))
        self.dir = os.path.join(root, task)
        self.meta_path = os.path.join(self.dir, "meta.json")
        self._meta_lock = threading.RLock()
        os.makedirs(self.dir, exist_ok=True)
        self._ensure_meta()

    # -- 元数据 -----------------------------------------------------------
    def _ensure_meta(self) -> dict:
        if not os.path.exists(self.meta_path):
            _atomic_write_json(self.meta_path, {
                "task": self.task,
                "shard_size": self.shard_size,
                "shard_count": 0,
                "total": 0,
                "next_id": 1,
                "created_at": time.time(),
            })
        return _read_json(self.meta_path, {})

    def _read_meta(self) -> dict:
        return _read_json(self.meta_path, {
            "task": self.task, "shard_size": self.shard_size,
            "shard_count": 0, "total": 0, "next_id": 1,
        })

    def _write_meta(self, meta: dict) -> None:
        _atomic_write_json(self.meta_path, meta)

    def _shard_path(self, index: int) -> str:
        return os.path.join(self.dir, f"shard_{index:06d}.json")

    # -- 读写原语 ---------------------------------------------------------
    def _read_shard(self, index: int) -> list:
        path = self._shard_path(index)
        if not os.path.exists(path):
            return []
        data = _read_json(path, [])
        return data if isinstance(data, list) else []

    def _write_shard(self, index: int, records: list) -> None:
        _atomic_write_json(self._shard_path(index), records)

    # -- 插入 -------------------------------------------------------------
    def insert(self, record: dict) -> str:
        """插入一条记录，返回其 id。

        排他锁保护「读取最后一个分片 -> 追加 -> 写回 -> 更新元数据」这个
        读-改-写序列，保证并发安全。
        """
        if not isinstance(record, dict):
            raise TypeError("record 必须是 dict")
        record = dict(record)
        with FileLock(lock_path_for(self.meta_path)):
            meta = self._read_meta()
            record_id = record.get("id") or f"{self.task}_{meta['next_id']}"
            meta["next_id"] = meta["next_id"] + 1
            record["id"] = record_id
            record.setdefault("_created", time.time())

            index = meta["shard_count"] - 1 if meta["shard_count"] else -1
            if index < 0:
                index = 0
                meta["shard_count"] = 1
                records = [record]
                self._write_shard(index, records)
            else:
                with FileLock(lock_path_for(self._shard_path(index))):
                    records = self._read_shard(index)
                    if len(records) >= self.shard_size:
                        index += 1
                        meta["shard_count"] = index + 1
                        records = [record]
                    else:
                        records = records + [record]
                    self._write_shard(index, records)

            meta["total"] = meta["total"] + 1
            self._write_meta(meta)
            return record_id

    def insert_many(self, records: Iterable[dict]) -> list[str]:
        """批量插入（事务式：要么全部成功，要么抛出）。"""
        ids: list[str] = []
        with FileLock(lock_path_for(self.meta_path)):
            meta = self._read_meta()
            pending = []
            for record in records:
                record = dict(record)
                rid = record.get("id") or f"{self.task}_{meta['next_id']}"
                meta["next_id"] += 1
                record["id"] = rid
                record.setdefault("_created", time.time())
                pending.append(record)
                ids.append(rid)

            index = meta["shard_count"] - 1 if meta["shard_count"] else -1
            if index < 0:
                index = 0
                meta["shard_count"] = 1
            # 逐个填充分片
            while pending:
                path = self._shard_path(index)
                with FileLock(lock_path_for(path)):
                    records = self._read_shard(index)
                    room = self.shard_size - len(records)
                    take = pending[:room]
                    self._write_shard(index, records + take)
                pending = pending[room:] if room else pending
                if pending:
                    index += 1
                    meta["shard_count"] = index + 1
            meta["total"] += len(ids)
            self._write_meta(meta)
        return ids

    # -- 查询 -------------------------------------------------------------
    def _iter_all_locked(self) -> Iterator[dict]:
        meta = self._read_meta()
        for index in range(meta.get("shard_count", 0)):
            for record in self._read_shard(index):
                yield record

    def all(self) -> list[dict]:
        """返回所有记录（按分片顺序）。"""
        with FileLock(lock_path_for(self.meta_path), mode="shared"):
            return list(self._iter_all_locked())

    def get(self, record_id: str) -> Optional[dict]:
        with FileLock(lock_path_for(self.meta_path), mode="shared"):
            for record in self._iter_all_locked():
                if record.get("id") == record_id:
                    return record
        return None

    def query(self, where: Optional[list] = None,
              order_by: Optional[str] = None,
              order: str = "asc",
              limit: Optional[int] = None,
              offset: int = 0) -> list[dict]:
        """跨分片查询并合并结果。

        ``where`` 形如 ``[("field", "op", value), ...]``，op 见 ``_OPERATORS``。
        过滤在扫描过程中进行，最终按 ``order_by`` 排序并分页。
        """
        conditions = where or []
        matched = []
        with FileLock(lock_path_for(self.meta_path), mode="shared"):
            for record in self._iter_all_locked():
                if self._matches(record, conditions):
                    matched.append(record)

        if order_by is not None:
            reverse = order == "desc"
            matched.sort(
                key=lambda r: r.get(order_by, ""),
                reverse=reverse,
            )
        if offset:
            matched = matched[offset:]
        if limit is not None:
            matched = matched[:limit]
        return matched

    @staticmethod
    def _matches(record: dict, conditions: list) -> bool:
        for field, op, value in conditions:
            actual = record
            for part in field.split("."):
                if not isinstance(actual, dict):
                    actual = None
                    break
                actual = actual.get(part)
            func = _OPERATORS.get(op)
            if func is None:
                raise ValueError(f"未知操作符: {op}")
            try:
                if not func(actual, value):
                    return False
            except (TypeError, ValueError):
                return False
        return True

    # -- 删除（墓碑） -----------------------------------------------------
    def delete(self, record_id: str) -> bool:
        with FileLock(lock_path_for(self.meta_path)):
            meta = self._read_meta()
            for index in range(meta.get("shard_count", 0)):
                path = self._shard_path(index)
                with FileLock(lock_path_for(path)):
                    records = self._read_shard(index)
                    for i, record in enumerate(records):
                        if record.get("id") == record_id:
                            records[i] = {"id": record_id, "_deleted": True}
                            self._write_shard(index, records)
                            meta["total"] -= 1
                            self._write_meta(meta)
                            return True
        return False

    # -- 合并 / 压缩 ------------------------------------------------------
    def compact(self) -> dict:
        """把所有存活记录重新写入尽可能少的分片，并清理墓碑。

        返回统计信息。合并过程中使用排他锁阻止并发写。
        """
        with FileLock(lock_path_for(self.meta_path)):
            live = [r for r in self._iter_all_locked() if not r.get("_deleted")]
            meta = self._read_meta()
            old_count = meta.get("shard_count", 0)
            new_count = (len(live) + self.shard_size - 1) // self.shard_size

            for index in range(old_count):
                path = self._shard_path(index)
                if index < new_count:
                    start = index * self.shard_size
                    chunk = live[start:start + self.shard_size]
                    self._write_shard(index, chunk)
                else:
                    # 多余的旧分片直接删除文件
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass

            meta["shard_count"] = new_count
            meta["total"] = len(live)
            self._write_meta(meta)
            return {
                "task": self.task,
                "before_shards": old_count,
                "after_shards": new_count,
                "records": len(live),
            }

    def merge(self) -> dict:
        """把全部分片归并为一个整体导出（不改变存储布局）。

        返回一个 dict，包含合并后的所有记录和统计信息，
        可用于导出或给查询层做一次性读取。
        """
        records = self.all()
        return {
            "task": self.task,
            "total": len(records),
            "shards": self._read_meta().get("shard_count", 0),
            "records": records,
        }

    def stats(self) -> dict:
        meta = self._read_meta()
        return {
            "task": self.task,
            "shard_count": meta.get("shard_count", 0),
            "total": meta.get("total", 0),
            "shard_size": meta.get("shard_size", self.shard_size),
        }


class StoreRegistry:
    """按任务名管理多个 :class:`ShardedStore` 的注册表。

    进程内缓存实例，避免重复创建；不同任务天然隔离到不同目录。
    """

    def __init__(self, root: str, shard_size: int = DEFAULT_SHARD_SIZE):
        self.root = root
        self.shard_size = shard_size
        self._stores: dict[str, ShardedStore] = {}
        self._lock = threading.Lock()

    def task(self, name: str) -> ShardedStore:
        with self._lock:
            if name not in self._stores:
                self._stores[name] = ShardedStore(self.root, name, self.shard_size)
            return self._stores[name]

    def tasks(self) -> list[str]:
        """返回所有实际存在（含 meta.json）的分片存储任务目录。"""
        if not os.path.isdir(self.root):
            return []
        result = []
        for d in os.listdir(self.root):
            full = os.path.join(self.root, d)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, "meta.json")):
                result.append(d)
        return sorted(result)

    def compact_all(self) -> list[dict]:
        return [self.task(t).compact() for t in self.tasks()]
