"""YAML loader helpers with `!include` support.

Manifests can get large; `!include` enables composing them from reusable YAML
fragments while still validating the final expanded payload via Pydantic.

Example:

```yaml
topology:
  layers:
    - !include ../blocks/attention/standard_opgraph.yml
    - !include
        path: ../blocks/attention/standard_opgraph.yml
        vars:
          qkv_dim: 6144
```

Includes are resolved relative to the including file and are restricted to the
nearest project root (directory containing `pyproject.toml` or `.git`) to avoid
accidental traversal outside the repo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from caramba.config.resolve import Resolver


def _find_project_root(start: Path) -> Path:
    p = start.resolve()
    for d in (p, *p.parents):
        if (d / "pyproject.toml").exists() or (d / ".git").exists():
            return d
    return p


def _is_within(path: Path, root: Path) -> bool:
    try:
        _ = path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


class _IncludeLoader(yaml.SafeLoader):
    """SafeLoader carrying include context (base_dir/root/seen)."""


def _construct_include(loader: _IncludeLoader, node: yaml.Node) -> object:
    rel: str | None = None
    local_vars: dict[str, object] | None = None

    if isinstance(node, yaml.ScalarNode):
        rel = loader.construct_scalar(node)
        if not isinstance(rel, str) or not rel.strip():
            raise ValueError("!include path must be a non-empty string")
    elif isinstance(node, yaml.MappingNode):
        raw = loader.construct_mapping(node, deep=True)  # type: ignore[arg-type]
        if not isinstance(raw, dict):
            raise TypeError("!include mapping payload must be a mapping")
        allowed = {"path", "vars"}
        raw_keys = {str(k) for k in raw.keys()}
        extra = sorted(raw_keys - allowed)
        if extra:
            raise ValueError(f"!include supports only keys {sorted(allowed)}, got extra keys {extra}")
        rel = raw.get("path", None)
        if not isinstance(rel, str) or not rel.strip():
            raise ValueError("!include mapping requires non-empty 'path' string")
        v = raw.get("vars", None)
        if v is not None:
            if not isinstance(v, dict):
                raise TypeError("!include 'vars' must be a mapping")
            local_vars = {str(k): v[k] for k in list(v.keys())}
    else:
        raise TypeError("!include expects a scalar path or a {path, vars} mapping")

    base_dir: Path = getattr(loader, "_base_dir")  # type: ignore[assignment]
    root_dir: Path = getattr(loader, "_root_dir")  # type: ignore[assignment]
    seen: set[Path] = getattr(loader, "_seen")  # type: ignore[assignment]

    inc = (base_dir / rel).resolve()
    if not _is_within(inc, root_dir):
        raise ValueError(f"!include path escapes project root: {rel!r}")
    if inc in seen:
        chain = " -> ".join(str(p) for p in list(seen) + [inc])
        raise ValueError(f"!include cycle detected: {chain}")

    payload = load_yaml(inc, _root_dir=root_dir, _seen=seen)
    if local_vars:
        payload = Resolver(local_vars, allow_unknown=True).resolve(payload)
    return payload


_IncludeLoader.add_constructor("!include", _construct_include)


def load_yaml(path: Path, *, _root_dir: Path | None = None, _seen: set[Path] | None = None) -> object:
    """Load a YAML file with `!include` expansion."""
    path = path.resolve()
    root_dir = _root_dir if _root_dir is not None else _find_project_root(path.parent)
    seen = _seen if _seen is not None else set()
    if path in seen:
        chain = " -> ".join(str(p) for p in list(seen) + [path])
        raise ValueError(f"!include cycle detected: {chain}")
    seen.add(path)

    text = path.read_text(encoding="utf-8")
    loader = _IncludeLoader(text)
    setattr(loader, "_base_dir", path.parent)
    setattr(loader, "_root_dir", root_dir)
    setattr(loader, "_seen", seen)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()
        seen.remove(path)
