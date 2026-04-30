from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

import opndet.primitives  # noqa: F401  registers primitives
from opndet.blocks import CSPBlock, DWSep, _conv_bn_act
from opndet.registry import get, register


@register("ConvBnAct")
class ConvBnAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, g: int = 1):
        super().__init__()
        self.block = _conv_bn_act(in_ch, out_ch, k=k, s=s, g=g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


register("CSPBlock")(CSPBlock)
register("DWSep")(DWSep)


def _resolve_from(f: int | str, aliases: dict[str, int], n_layers_so_far: int) -> int:
    if isinstance(f, str):
        if f not in aliases:
            raise KeyError(f"Alias '{f}' not defined before use")
        return aliases[f]
    if f < 0:
        return n_layers_so_far + f + 1
    return f


def _build_module(spec: dict[str, Any]) -> nn.Module:
    cls = get(spec["module"])
    return cls(**(spec.get("args") or {}))


class YamlModel(nn.Module):
    def __init__(
        self,
        modules: nn.ModuleList,
        graph: list,
        multi_input: list[bool],
        output_specs: list[dict[str, Any]],
        input_shape: tuple[int, int, int],
        aliases: dict[str, int] | None = None,
    ):
        super().__init__()
        self.layers = modules
        self._graph = graph
        self._multi = multi_input
        self._out_specs = output_specs
        self.input_shape = input_shape
        self.aliases = aliases or {}

    def _run(self, x: torch.Tensor) -> list[torch.Tensor]:
        cache: list[torch.Tensor] = [x]
        for mod, frm, multi in zip(self.layers, self._graph, self._multi):
            if multi:
                inp = [cache[j] for j in frm]
                out = mod(inp)
            else:
                out = mod(cache[frm])
            cache.append(out)
        return cache

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        cache = self._run(x)
        result: dict[str, torch.Tensor] = {}
        for spec in self._out_specs:
            t = cache[spec["layer_idx"]]
            s, e = spec["start"], spec["end"]
            if s is not None:
                t = t[:, s:e]
            act: nn.Module | None = spec["activation_fn"]
            if act is not None:
                t = act(t)
            result[spec["name"]] = t
        return result

    def forward_with_alias(self, x: torch.Tensor, alias: str) -> torch.Tensor:
        if alias not in self.aliases:
            raise KeyError(f"alias '{alias}' not found. available: {sorted(self.aliases)}")
        cache = self._run(x)
        return cache[self.aliases[alias]]


def _parse_activation(act_name: str | None) -> nn.Module | None:
    if act_name is None or act_name == "none":
        return None
    from opndet.primitives import ACTIVATIONS

    if act_name not in ACTIVATIONS:
        raise KeyError(f"Unknown activation '{act_name}'. Available: {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[act_name]()


def build_model_from_yaml(path: str | Path) -> YamlModel:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    spec = cfg["model"]
    in_ch = spec.get("in_ch", 3)
    img_h = spec.get("img_h", 384)
    img_w = spec.get("img_w", 512)
    layer_specs: list[dict] = spec["layers"]
    output_cfg: list[dict] = spec["outputs"]

    aliases: dict[str, int] = {}
    modules: list[nn.Module] = []
    graph: list = []
    multi_input: list[bool] = []

    for i, ls in enumerate(layer_specs):
        raw_from = ls.get("from", -1)
        if isinstance(raw_from, list):
            graph.append([_resolve_from(f, aliases, i) for f in raw_from])
            multi_input.append(True)
        else:
            graph.append(_resolve_from(raw_from, aliases, i))
            multi_input.append(False)

        modules.append(_build_module(ls))

        if "name" in ls:
            aliases[ls["name"]] = i + 1

    n = len(layer_specs)
    output_specs: list[dict] = []
    for out in output_cfg:
        raw_from = out.get("from", -1)
        layer_idx = _resolve_from(raw_from, aliases, n)
        channels = out.get("channels")
        start, end = (channels[0], channels[1]) if channels is not None else (None, None)
        output_specs.append(
            {
                "name": out["name"],
                "layer_idx": layer_idx,
                "start": start,
                "end": end,
                "activation_fn": _parse_activation(out.get("activation")),
            }
        )

    return YamlModel(
        modules=nn.ModuleList(modules),
        graph=graph,
        multi_input=multi_input,
        output_specs=output_specs,
        input_shape=(in_ch, img_h, img_w),
        aliases=aliases,
    )
