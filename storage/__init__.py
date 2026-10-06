"""存储层：分片 JSON 存储 + 文件锁 + 相似度索引服务。"""

from .lock import FileLock, LockTimeout, lock_path_for
from .sharded import ShardedStore, StoreRegistry, _atomic_write_json, _read_json
from .similarity_store import (DEFAULT_DUP_THRESHOLD, DEFAULT_LEX_FLOOR,
                               SimilarityService)

__all__ = [
    "FileLock",
    "LockTimeout",
    "lock_path_for",
    "ShardedStore",
    "StoreRegistry",
    "SimilarityService",
    "DEFAULT_DUP_THRESHOLD",
    "DEFAULT_LEX_FLOOR",
    "_atomic_write_json",
    "_read_json",
]
