"""流水线编排引擎。

难点聚焦：
1. **DAG 编排**：阶段之间可以配置依赖，引擎做拓扑排序、环检测，
   无依赖的阶段并行执行（ThreadPoolExecutor）。
2. **数据流**：阶段共享一个上下文 dict，按声明读写指定字段，
   未声明依赖时根据 ``outputs -> inputs`` 自动推导依赖边。
3. **批量处理**：多文档并行跑同一流水线，支持分块与进度回调，
   结果与错误分离返回，供上层做分片持久化。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Optional

from .stage import Stage


class PipelineError(RuntimeError):
    def __init__(self, message: str, errors: Optional[dict] = None):
        super().__init__(message)
        self.errors = errors or {}


class Pipeline:
    """由若干阶段组成的有向无环图。"""

    def __init__(self, name: str, stages: dict[str, Stage],
                 deps: Optional[dict[str, list[str]]] = None):
        self.name = name
        self.stages = stages
        self.deps = self._resolve_deps(deps or {})
        self.order = self._topological_sort()

    # -- 依赖推导 ---------------------------------------------------------
    def _resolve_deps(self, explicit: dict[str, list[str]]) -> dict[str, list[str]]:
        """合并显式依赖与按 output->input 自动推导的依赖。"""
        resolved = {name: list(deps) for name, deps in explicit.items()}
        produced_by: dict[str, str] = {}
        for name, stage in self.stages.items():
            for out in stage.outputs:
                produced_by[out] = name

        for name, stage in self.stages.items():
            for inp in stage.inputs:
                producer = produced_by.get(inp)
                if producer and producer != name and producer not in resolved.get(name, []):
                    resolved.setdefault(name, []).append(producer)
        return resolved

    # -- 拓扑排序（Kahn） -------------------------------------------------
    def _topological_sort(self) -> list[str]:
        indegree = {name: 0 for name in self.stages}
        dependents: dict[str, list[str]] = {name: [] for name in self.stages}
        for name, deps in self.deps.items():
            for d in deps:
                if d not in self.stages:
                    raise PipelineError(f"阶段 '{name}' 依赖了不存在的阶段 '{d}'")
                indegree[name] += 1
                dependents[d].append(name)

        queue = [n for n in self.stages if indegree[n] == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for dep in dependents[node]:
                indegree[dep] -= 1
                if indegree[dep] == 0:
                    queue.append(dep)

        if len(order) != len(self.stages):
            remaining = [n for n in self.stages if n not in order]
            raise PipelineError(f"流水线存在环，无法执行的阶段: {remaining}")
        return order

    # -- 执行 -------------------------------------------------------------
    def run(self, ctx: dict, params: Optional[dict] = None,
            max_workers: Optional[int] = None) -> dict:
        """执行流水线，返回最终上下文。

        按拓扑顺序推进，每一「批」就绪阶段并行执行。
        """
        remaining = set(self.order)
        executed: set = set()
        errors: dict[str, str] = {}

        while remaining:
            ready = [n for n in remaining
                     if all(d in executed for d in self.deps.get(n, []))]
            if not ready:
                raise PipelineError("存在环或依赖缺失", errors)
            with ThreadPoolExecutor(
                max_workers=max_workers or max(1, len(ready))
            ) as ex:
                futures = {
                    ex.submit(self.stages[n].run, ctx, params): n for n in ready
                }
                for fut in as_completed(futures):
                    name = futures[fut]
                    try:
                        fut.result()
                        executed.add(name)
                    except Exception as exc:  # noqa: BLE001
                        errors[name] = f"{type(exc).__name__}: {exc}"
                        executed.add(name)
            remaining -= executed

        if errors:
            raise PipelineError(f"{len(errors)} 个阶段执行失败", errors)
        return ctx

    def describe(self) -> dict:
        return {
            "name": self.name,
            "order": self.order,
            "deps": self.deps,
            "stages": [self.stages[n].describe() for n in self.order],
        }


class PipelineEngine:
    """阶段注册表 + 流水线构建 + 批量执行。"""

    def __init__(self):
        self.registry: dict[str, Stage] = {}

    def register(self, stage: Stage) -> None:
        self.registry[stage.name] = stage

    def register_many(self, stages: Iterable[Stage]) -> None:
        for stage in stages:
            self.register(stage)

    def register_builtin(self) -> "PipelineEngine":
        from .stages import BUILTIN_STAGES
        self.register_many(BUILTIN_STAGES)
        return self

    # -- 构建 -------------------------------------------------------------
    def build(self, config: dict) -> Pipeline:
        """从配置构建流水线。

        config 形如::

            {
              "name": "demo",
              "stages": [
                {"name": "segment", "params": {...}, "deps": []},
                {"name": "pos", "deps": ["segment"]},
              ]
            }
        """
        name = config.get("name", "pipeline")
        stages: dict[str, Stage] = {}
        deps: dict[str, list[str]] = {}
        for item in config.get("stages", []):
            sname = item["name"]
            if sname not in self.registry:
                raise PipelineError(f"未注册的阶段: {sname}")
            stage = self.registry[sname]
            stages[sname] = stage
            if item.get("params"):
                # 用参数包装生成新阶段实例（默认参数被覆盖）
                stage = Stage(
                    name=sname, func=stage.func,
                    inputs=stage.inputs, outputs=stage.outputs,
                    description=stage.description, params=item["params"],
                )
                stages[sname] = stage
            if item.get("deps"):
                deps[sname] = item["deps"]
        if not stages:
            raise PipelineError("流水线没有阶段")
        return Pipeline(name, stages, deps)

    # -- 批量执行 ---------------------------------------------------------
    def run_batch(self, config: dict, documents: list[str],
                  shared: Optional[dict] = None,
                  max_workers: int = 4,
                  chunk_size: int = 32,
                  progress: Optional[Callable[[int, int], None]] = None) -> list[dict]:
        """对多篇文档并行执行同一流水线。

        每篇文档在独立的上下文副本中处理，结果包含文档索引、输出与错误，
        保证单篇失败不影响其它文档。
        """
        pipeline = self.build(config)
        shared = shared or {}
        results: list[Optional[dict]] = [None] * len(documents)

        def _run_one(idx_text):
            idx, text = idx_text
            ctx = dict(shared)
            ctx["text"] = text
            ctx["doc_index"] = idx
            try:
                out = pipeline.run(ctx, max_workers=1)
                return idx, {"ok": True, "output": out}
            except Exception as exc:  # noqa: BLE001
                return idx, {"ok": False, "error": str(exc), "output": ctx}

        done = 0
        total = len(documents)
        # 分块处理，控制并发与内存峰值
        for start in range(0, total, chunk_size):
            chunk = list(enumerate(documents[start:start + chunk_size], start=start))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_run_one, item) for item in chunk]
                for fut in as_completed(futures):
                    idx, res = fut.result()
                    results[idx] = res
                    done += 1
            if progress:
                progress(done, total)

        return results

    def list_stages(self) -> list[dict]:
        return [s.describe() for s in self.registry.values()]
