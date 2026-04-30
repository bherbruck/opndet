from __future__ import annotations

from typing import Type

import torch.nn as nn

_REGISTRY: dict[str, Type[nn.Module]] = {}


def register(name: str | None = None):
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        key = name if name is not None else cls.__name__
        _REGISTRY[key] = cls
        return cls

    return decorator


def get(name: str) -> Type[nn.Module]:
    if name not in _REGISTRY:
        raise KeyError(f"Block '{name}' not in registry. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
