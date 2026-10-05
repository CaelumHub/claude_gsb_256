"""流水线阶段定义。"""

from __future__ import annotations

from typing import Callable, Optional


class Stage:
    """流水线中的一个处理阶段。

    :param name: 唯一名称
    :param func: 处理函数 ``func(ctx, params)``，可返回 dict 合并进上下文
    :param inputs: 依赖的上下文字段（用于自动推导依赖关系）
    :param outputs: 产出的上下文字段
    :param description: 说明
    :param params: 默认参数
    """

    def __init__(self, name: str, func: Callable,
                 inputs: Optional[list[str]] = None,
                 outputs: Optional[list[str]] = None,
                 description: str = "",
                 params: Optional[dict] = None):
        self.name = name
        self.func = func
        self.inputs = inputs or []
        self.outputs = outputs or [name]
        self.description = description
        self.params = params or {}

    def run(self, ctx: dict, params: Optional[dict] = None) -> dict:
        merged = dict(self.params)
        if params:
            merged.update(params)
        result = self.func(ctx, merged)
        if isinstance(result, dict):
            ctx.update(result)
        return ctx

    def describe(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "params": self.params,
        }
