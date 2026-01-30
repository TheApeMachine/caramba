from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import yaml

from benchmark.behavior.schema import (
    BehaviorSuiteSpec,
    ChoiceExplicitSpec,
    ChoiceFromPoolSpec,
)
from benchmark.behavior.types import GeneratedCase, MatchType, EvalKind


_SLOT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _render_template(*, s: str, slots: dict[str, str]) -> str:
    """Render ${slot} placeholders. Strict: fail on missing slots or leftover placeholders."""

    def _sub(m: re.Match[str]) -> str:
        k = str(m.group(1))
        if k not in slots:
            raise KeyError(f"Template placeholder {k!r} has no generated slot value.")
        return str(slots[k])

    out = _SLOT_RE.sub(_sub, str(s))
    # Strict: no leftover placeholders.
    if _SLOT_RE.search(out) is not None:
        raise ValueError(f"Unresolved template placeholders remain in: {out!r}")
    return out


def _apply_transform(v: str, transform: str) -> str:
    t = str(transform or "none").lower()
    if t == "none":
        return str(v)
    if t == "upper":
        return str(v).upper()
    if t == "lower":
        return str(v).lower()
    if t == "title":
        return str(v).title()
    raise ValueError(f"Unknown slot transform: {transform!r}")


def _calc_value(
    *,
    cat_id: str,
    tmpl_id: str,
    slot_name: str,
    calc: dict[str, Any],
    slots: dict[str, str],
) -> str:
    """Compute a derived slot value from prior slots.

    Supported ops:
    - add/sub/mul: integer arithmetic on two args
    - concat: string concatenation of args
    - reverse: reverse string of single arg
    - repeat: repeat string arg N times (requires `n`)
    """
    if not isinstance(calc, dict):
        raise TypeError(f"{cat_id}:{tmpl_id}:{slot_name}: calc must be a dict.")
    op = str(calc.get("op", "")).strip().lower()
    args = calc.get("args", [])
    if not isinstance(args, list) or not args:
        raise ValueError(f"{cat_id}:{tmpl_id}:{slot_name}: calc.args must be a non-empty list.")

    def _get(a: Any) -> str:
        key = str(a)
        if key not in slots:
            raise KeyError(
                f"{cat_id}:{tmpl_id}:{slot_name}: calc references {key!r} which is not defined yet."
            )
        return str(slots[key])

    if op in {"add", "sub", "mul"}:
        if len(args) != 2:
            raise ValueError(f"{cat_id}:{tmpl_id}:{slot_name}: op={op} requires exactly 2 args.")
        x = int(_get(args[0]))
        y = int(_get(args[1]))
        if op == "add":
            return str(x + y)
        if op == "sub":
            return str(x - y)
        return str(x * y)

    if op in {"gt", "lt", "eq"}:
        if len(args) != 2:
            raise ValueError(f"{cat_id}:{tmpl_id}:{slot_name}: op={op} requires exactly 2 args.")
        x = int(_get(args[0]))
        y = int(_get(args[1]))
        if op == "gt":
            return "true" if x > y else "false"
        if op == "lt":
            return "true" if x < y else "false"
        return "true" if x == y else "false"

    if op == "concat":
        return "".join(_get(a) for a in args)

    if op == "reverse":
        if len(args) != 1:
            raise ValueError(f"{cat_id}:{tmpl_id}:{slot_name}: op=reverse requires exactly 1 arg.")
        return _get(args[0])[::-1]

    if op == "repeat":
        if len(args) != 1:
            raise ValueError(f"{cat_id}:{tmpl_id}:{slot_name}: op=repeat requires exactly 1 arg.")
        n = calc.get("n", None)
        if n is None:
            raise ValueError(f"{cat_id}:{tmpl_id}:{slot_name}: op=repeat requires field `n`.")
        nn = int(n)
        if nn < 0:
            raise ValueError(f"{cat_id}:{tmpl_id}:{slot_name}: repeat n must be >= 0.")
        return _get(args[0]) * nn

    raise ValueError(f"{cat_id}:{tmpl_id}:{slot_name}: unknown calc op {op!r}")


def load_behavior_suite(
    path: str | Path, *, seed_override: int | None = None
) -> tuple[BehaviorSuiteSpec, list[GeneratedCase]]:
    """Load suite YAML and generate concrete cases (deterministic, strict)."""

    p = Path(path)
    payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    spec = BehaviorSuiteSpec.model_validate(payload)
    if seed_override is not None:
        spec.seed = int(seed_override)

    rng = random.Random(int(spec.seed))
    cases: list[GeneratedCase] = []

    for cat in spec.categories:
        cat_cases: list[GeneratedCase] = []

        for tmpl in cat.templates:
            # Validate template-kind vs choice config.
            if tmpl.kind == EvalKind.CHOICE_LOGPROB and tmpl.choice is None:
                raise ValueError(f"{cat.id}:{tmpl.id}: kind=choice_logprob requires `choice` config.")
            if tmpl.kind != EvalKind.CHOICE_LOGPROB and tmpl.choice is not None:
                raise ValueError(f"{cat.id}:{tmpl.id}: `choice` is only valid for kind=choice_logprob.")

            for rep_i in range(int(tmpl.repeat)):
                # Generate slots for this instance.
                slots: dict[str, str] = {}
                # Per-instance uniqueness book-keeping.
                local_used: dict[str, set[str]] = {}

                for slot_name, slot_spec in (tmpl.slots or {}).items():
                    v: str
                    if slot_spec.pool is not None:
                        pool_name = str(slot_spec.pool)
                        if pool_name not in spec.pools:
                            raise KeyError(
                                f"{cat.id}:{tmpl.id}: unknown pool {pool_name!r} for slot {slot_name!r}."
                            )
                        pool = list(spec.pools[pool_name])
                        if not pool:
                            raise ValueError(f"{cat.id}:{tmpl.id}: pool {pool_name!r} is empty.")

                        uniq = str(slot_spec.unique_key) if slot_spec.unique_key else None
                        if uniq is not None:
                            local_used.setdefault(uniq, set())

                        if uniq is None:
                            v = str(rng.choice(pool))
                        else:
                            banned = set(local_used[uniq])
                            choices = [x for x in pool if str(x) not in banned]
                            if not choices:
                                raise ValueError(
                                    f"{cat.id}:{tmpl.id}: cannot satisfy uniqueness for key {uniq!r} "
                                    f"(pool={pool_name!r}, banned={len(banned)})."
                                )
                            v = str(rng.choice(choices))
                            local_used[uniq].add(v)

                    elif slot_spec.ref is not None:
                        ref = str(slot_spec.ref)
                        if ref not in slots:
                            raise KeyError(
                                f"{cat.id}:{tmpl.id}: slot {slot_name!r} references {ref!r} "
                                f"which is not defined yet (define referenced slots earlier)."
                            )
                        v = str(slots[ref])

                    elif slot_spec.calc is not None:
                        v = _calc_value(
                            cat_id=str(cat.id),
                            tmpl_id=str(tmpl.id),
                            slot_name=str(slot_name),
                            calc=slot_spec.calc,
                            slots=slots,
                        )
                    else:
                        raise ValueError(f"{cat.id}:{tmpl.id}: slot {slot_name!r} has no source.")

                    slots[str(slot_name)] = _apply_transform(v, str(slot_spec.transform))

                prompt = _render_template(s=str(tmpl.prompt), slots=slots)
                expected = _render_template(s=str(tmpl.expected), slots=slots)

                target_text = tmpl.target_text
                if target_text is None or not str(target_text).strip():
                    target_text_s = expected
                else:
                    target_text_s = _render_template(s=str(target_text), slots=slots)

                choices: list[str] = []
                correct_index: int | None = None
                if tmpl.choice is not None:
                    if isinstance(tmpl.choice, ChoiceExplicitSpec):
                        choices = [_render_template(s=str(c), slots=slots) for c in list(tmpl.choice.choices)]
                        if not choices:
                            raise ValueError(f"{cat.id}:{tmpl.id}: choice.explicit has no choices.")
                    elif isinstance(tmpl.choice, ChoiceFromPoolSpec):
                        pool_name = str(tmpl.choice.pool)
                        if pool_name not in spec.pools:
                            raise KeyError(f"{cat.id}:{tmpl.id}: unknown choice pool {pool_name!r}.")
                        correct = _render_template(s=str(tmpl.choice.correct), slots=slots)
                        pool = [str(x) for x in list(spec.pools[pool_name])]
                        # Normalize exclusion: if correct has leading whitespace, exclude both variants.
                        correct_raw = str(correct).lstrip() if str(correct).startswith(" ") else str(correct)
                        pool = [x for x in pool if (str(x) != str(correct) and str(x) != str(correct_raw))]
                        need = int(tmpl.choice.num_choices) - 1
                        if need < 1:
                            raise ValueError(f"{cat.id}:{tmpl.id}: num_choices must be >= 2.")
                        if len(pool) < need:
                            raise ValueError(
                                f"{cat.id}:{tmpl.id}: choice pool {pool_name!r} too small "
                                f"(need {need} distractors excluding correct)."
                            )
                        distractors = rng.sample(pool, k=need)
                        # If correct begins with a leading space, mirror that for distractors.
                        if str(correct).startswith(" "):
                            distractors = [" " + str(d).lstrip() for d in distractors]
                        choices = [correct] + distractors
                    else:
                        raise TypeError(f"{cat.id}:{tmpl.id}: unsupported choice spec type: {type(tmpl.choice)!r}")

                    # Shuffle choices if configured.
                    shuffle = bool(getattr(tmpl.choice, "shuffle", True))
                    if shuffle:
                        rng.shuffle(choices)

                    # Determine correct index (expected must match one and only one choice).
                    hits = [i for i, c in enumerate(choices) if str(c) == str(expected)]
                    if len(hits) != 1:
                        raise ValueError(
                            f"{cat.id}:{tmpl.id}: expected must match exactly one choice after rendering/shuffle "
                            f"(expected={expected!r}, hits={hits}, choices={choices!r})."
                        )
                    correct_index = int(hits[0])

                # Stable id within suite.
                cid = f"{cat.id}_{tmpl.id}_{rep_i}"
                cat_cases.append(
                    GeneratedCase(
                        id=str(cid),
                        category=str(cat.id),
                        difficulty=tmpl.difficulty,
                        kind=tmpl.kind,
                        prompt=str(prompt),
                        expected=str(expected),
                        choices=[str(x) for x in choices],
                        correct_index=correct_index,
                        allow_contained=bool(tmpl.allow_contained),
                        contained_constraints=list(tmpl.contained_constraints or []),
                        disallow_contained_if_expected_in_prompt=bool(
                            tmpl.disallow_contained_if_expected_in_prompt
                        ),
                        target_text=str(target_text_s) if str(target_text_s).strip() else None,
                        metadata=json.loads(json.dumps(tmpl.metadata or {})),
                    )
                )

        if len(cat_cases) != int(spec.tests_per_category):
            raise ValueError(
                f"Category {cat.id!r} expanded to {len(cat_cases)} cases, "
                f"but suite.tests_per_category={int(spec.tests_per_category)}."
            )

        cases.extend(cat_cases)

    # Strict: case ids must be unique across suite.
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Generated suite has duplicate case ids.")

    return spec, cases


def suite_snapshot(*, spec: BehaviorSuiteSpec) -> dict[str, Any]:
    """Stable, JSON-serializable snapshot of a spec (for embedding in artifacts)."""

    # Pydantic already has JSON-able fields; ensure no Enum objects remain.
    def _coerce(x: Any) -> Any:
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        if isinstance(x, list):
            return [_coerce(v) for v in x]
        if isinstance(x, dict):
            return {str(k): _coerce(v) for k, v in x.items()}
        # Enums / pydantic models
        try:
            if hasattr(x, "value"):
                return str(getattr(x, "value"))
        except Exception:
            pass
        try:
            if hasattr(x, "model_dump"):
                return _coerce(x.model_dump())
        except Exception:
            pass
        raise TypeError(f"Non-serializable suite field: {type(x).__name__}")

    return _coerce(spec.model_dump())

