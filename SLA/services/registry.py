"""Versioned dispatch from DSL model references to service plugins."""

from __future__ import annotations

from typing import Callable

from ..dsl.compiler import SLAProblem
from .commodity_delivery import compile_commodity_delivery


SERVICE_PLUGINS: dict[str, Callable] = {
    "commodity-delivery-v1": compile_commodity_delivery,
}


def compile_service(problem: SLAProblem):
    try:
        compiler = SERVICE_PLUGINS[problem.models.service]
    except KeyError as exc:
        raise ValueError(f"service plugin {problem.models.service!r} is not implemented") from exc
    return compiler(problem)
