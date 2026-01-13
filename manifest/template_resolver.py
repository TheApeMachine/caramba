"""Template resolver

Template manifests support string interpolation to keep large YAML files readable.

Supported placeholders:

- `${name}` resolves from the template variable map.
- `${ENV:NAME}` resolves from process environment variables.

Interpolation preserves types when the placeholder is the entire string value.
For example, `"${steps}"` becomes an integer when `steps` is an integer variable.
"""

from __future__ import annotations

import re
from collections.abc import Mapping


class TemplateResolver:
    """Resolver for template-time placeholders in manifests."""

    def __init__(
        self,
        vars: Mapping[str, object],
        *,
        env: Mapping[str, str],
        allow_unknown: bool = False,
    ) -> None:
        """Create a new resolver.

        Args:
            vars: Mapping of template variables used by `${name}`.
            env: Process environment mapping used by `${ENV:NAME}`.
            allow_unknown: If true, unresolved variables are left intact.
        """
        super().__init__()
        self.vars: dict[str, object] = dict(vars)
        self.env: Mapping[str, str] = env
        self.allow_unknown = bool(allow_unknown)
        self.cache: dict[str, object] = {}
        self.resolving: set[str] = set()
        self.pattern = re.compile(r"\$\{([A-Za-z0-9_]+|ENV:[A-Za-z0-9_]+)\}")

    def resolve(self, value: object) -> object:
        """Resolve placeholders within a payload node."""
        if isinstance(value, Mapping):
            return {k: self.resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self.resolve(v) for v in value)
        if isinstance(value, str):
            return self.resolve_str(value)
        return value

    def resolve_str(self, value: str) -> object:
        """Resolve placeholders inside a string value."""
        matches = list(self.pattern.finditer(value))
        if not matches:
            return value

        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            return self.resolve_key(matches[0].group(1))

        def replace(match: re.Match[str]) -> str:
            return str(self.resolve_key(match.group(1)))

        return self.pattern.sub(replace, value)

    def resolve_key(self, key: str) -> object:
        """Resolve a single `${key}` placeholder."""
        if key.startswith("ENV:"):
            env_name = key.split(":", 1)[1]
            if env_name in self.env:
                return self.env[env_name]
            if self.allow_unknown:
                return f"${{{key}}}"
            raise ValueError(f"Missing environment variable for placeholder: {key!r}")

        if key in self.cache:
            return self.cache[key]
        if key in self.resolving:
            raise ValueError(f"Cycle detected in manifest variables: {key}")
        if key not in self.vars:
            if self.allow_unknown:
                return f"${{{key}}}"
            raise ValueError(f"Unknown manifest variable: {key}")

        self.resolving.add(key)
        resolved = self.resolve(self.vars[key])
        self.resolving.remove(key)
        self.cache[key] = resolved
        return resolved

