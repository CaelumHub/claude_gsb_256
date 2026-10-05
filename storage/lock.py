"""跨进程文件锁。

基于 ``fcntl.flock`` 实现，用于保护 JSON 分片存储的并发读写。
写操作使用排他锁 (LOCK_EX)，读操作使用共享锁 (LOCK_SH)，
配合带超时的轮询等待，避免多进程/多线程同时写入导致的数据损坏。

难点说明：
    JSON 文件不像数据库有事务，多个 worker 同时追加写会互相覆盖。
    这里用「锁文件 + flock」把每个分片/元数据的读-改-写变成一个临界区，
    并通过「写临时文件 + os.replace 原子替换」保证崩溃时不会留下半截文件。
"""

from __future__ import annotations

import fcntl
import os
import time
from typing import Optional


class LockTimeout(RuntimeError):
    """等待文件锁超时。"""

    def __init__(self, path: str, timeout: float):
        self.path = path
        self.timeout = timeout
        super().__init__(
            f"无法在 {timeout:.1f}s 内获取文件锁: {path}"
        )


class FileLock:
    """基于 flock 的文件锁上下文管理器。

    用法::

        with FileLock("/data/task/meta.json.lock"):
            ...  # 临界区

    ``mode`` 为 ``"exclusive"`` 或 ``"shared"``。
    """

    def __init__(self, path: str, timeout: float = 10.0,
                 mode: str = "exclusive"):
        self.path = path
        self.timeout = timeout
        self.mode = mode
        self._fd: Optional[object] = None

    # -- 生命周期 ---------------------------------------------------------
    def acquire(self) -> "FileLock":
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # 以追加模式打开，保证锁文件本身不会因为其它进程写入而被截断
        self._fd = open(self.path, "a")
        operation = fcntl.LOCK_EX if self.mode == "exclusive" else fcntl.LOCK_SH
        deadline = time.monotonic() + self.timeout

        while True:
            try:
                fcntl.flock(self._fd.fileno(), operation | fcntl.LOCK_NB)
                return self
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    self.release()
                    raise LockTimeout(self.path, self.timeout)
                time.sleep(0.02)

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                self._fd.close()
            except OSError:
                pass
            self._fd = None

    # -- 上下文协议 -------------------------------------------------------
    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def lock_path_for(target: str) -> str:
    """给定数据文件路径，返回对应的锁文件路径。

    锁文件独立于数据文件，避免锁文件内容与数据混淆；
    但锁文件与数据文件同名前缀便于排查。
    """
    return target + ".lock"
