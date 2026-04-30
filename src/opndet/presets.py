from __future__ import annotations

from importlib import resources
from pathlib import Path


def list_presets() -> list[str]:
    files = resources.files("opndet.configs")
    return sorted(
        f.name.removeprefix("opndet-").removesuffix(".yaml")
        for f in files.iterdir()
        if f.name.startswith("opndet-") and f.name.endswith(".yaml")
    )


def resolve(preset_or_path: str) -> str:
    """Resolve 'bbox-s' to the bundled YAML path, or pass through a literal path."""
    p = Path(preset_or_path)
    if p.exists():
        return str(p)
    bundled = resources.files("opndet.configs").joinpath(f"opndet-{preset_or_path}.yaml")
    if bundled.is_file():
        with resources.as_file(bundled) as real_path:
            return str(real_path)
    raise FileNotFoundError(
        f"'{preset_or_path}' is neither an existing path nor a bundled preset. "
        f"Available presets: {list_presets()}"
    )


def bundled_train_template() -> str:
    """Return path to the bundled default training config template."""
    f = resources.files("opndet.configs").joinpath("train.yaml")
    with resources.as_file(f) as p:
        return str(p)
