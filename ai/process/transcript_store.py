"""Transcript store for interactive agent processes.

This module provides a single, composable object responsible for:
- storing conversation history as `google.genai.types.Content`
- persisting that history to a JSONL transcript
- enforcing hard context budgets (items + tokens)

The brainstorm/multiplex chat process uses this to guarantee prompts remain
bounded and reproducible from disk without hidden configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

from google.genai import types
import tiktoken


class TranscriptStore:
    """Transcript store with hard token and item budgets.

    Used to keep interactive chat history bounded so model prompts never exceed
    the configured context window, while retaining a durable JSONL log for
    reproducibility and post-hoc analysis.
    """

    def __init__(
        self,
        *,
        path: str,
        max_items: int,
        max_tokens: int,
        max_event_tokens: int,
        compact_after_bytes: int,
    ) -> None:
        self.path = Path(path)
        self.max_items = max_items
        self.max_tokens = max_tokens
        self.max_event_tokens = max_event_tokens
        self.compact_after_bytes = compact_after_bytes
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.history: list[types.Content] = []
        self.loaded = False

    def load(self) -> None:
        """Load transcript JSONL into memory.

        Raises:
            ValueError: If any line is not valid JSON or does not contain role/text.
        """
        if self.loaded:
            return
        self.loaded = True

        if not self.path.exists():
            return

        lines = self.path.read_text(encoding="utf-8").splitlines()
        for line_number, raw in enumerate(lines, start=1):
            if not raw.strip():
                continue
            self.history.append(self.parse_event_line(raw, line_number=line_number))

        self.trim_in_place()

    def parse_event_line(self, raw: str, *, line_number: int) -> types.Content:
        """Parse one JSONL line into a Content message.

        This supports both the current transcript schema:
          {"role": "...", "text": "..."}

        And legacy-compatible shapes:
          {"type": "...", "content": "..."}
          {"type": "...", "text": "..."}
          {"role": "...", "content": "..."}
          {"role": "...", "parts": [{"text": "..."}]}
        """
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            excerpt = raw[:200].replace("\n", "\\n")
            raise ValueError(
                f"Transcript line {line_number} is not valid JSON: {e}. "
                f"Line excerpt: {excerpt!r}. "
                f"Fix by deleting {self.path} or repairing it to valid JSONL."
            ) from e
        if not isinstance(obj, dict):
            raise ValueError(
                f"Transcript line {line_number} must be a JSON object, got {type(obj).__name__}. "
                f"Fix by deleting {self.path} (or rewriting it as JSONL objects)."
            )
        role, text = self.validate_event_object(obj, line_number=line_number)
        return types.Content(role=role, parts=[types.Part(text=text)])

    def validate_event_object(self, obj: dict[object, object], *, line_number: int) -> tuple[str, str]:
        """Validate transcript object shape and return (role, text)."""
        role = obj.get("role", None)
        text = obj.get("text", None)
        if isinstance(role, str) and role and isinstance(text, str) and text:
            return role, text

        legacy_type = obj.get("type", None)
        legacy_content = obj.get("content", None)
        if isinstance(legacy_type, str) and legacy_type:
            if isinstance(text, str) and text:
                return legacy_type, text
            if isinstance(legacy_content, str) and legacy_content:
                return legacy_type, legacy_content

        if isinstance(role, str) and role and isinstance(legacy_content, str) and legacy_content:
            return role, legacy_content

        parts = obj.get("parts", None)
        if isinstance(role, str) and role and isinstance(parts, list) and parts:
            texts: list[str] = []
            for part in parts:
                if isinstance(part, dict):
                    part_text = part.get("text", None)
                    if isinstance(part_text, str) and part_text:
                        texts.append(part_text)
            if texts:
                return role, "\n".join(texts)

        keys = ", ".join(sorted(str(k) for k in obj.keys()))
        raise ValueError(
            f"Transcript line {line_number} missing role/text fields. "
            f"Expected keys like (role,text) or (type,content). Found keys: {keys}. "
            f"Fix by deleting {self.path} or repairing it to include role/text per line."
        )

    def append_markdown_event(self, *, role: str, author: str, content: str) -> None:
        """Append a single markdown-formatted event to history and disk."""
        if not isinstance(role, str) or not role:
            raise ValueError("role must be a non-empty string")
        if not isinstance(author, str) or not author:
            raise ValueError("author must be a non-empty string")
        if not isinstance(content, str) or not content:
            raise ValueError("content must be a non-empty string")

        text = f"**{author}**: {self.truncate_to_tokens(content, self.max_event_tokens)}"
        event = types.Content(role=role, parts=[types.Part(text=text)])
        self.history.append(event)
        self.trim_in_place()
        self.persist_event({"role": role, "text": text})
        self.compact_file_if_needed()

    def append_event(self, *, role: str, text: str) -> None:
        """Append a plain (role,text) event to history and disk.

        This is the preferred format for durable chat transcripts that may be
        re-ingested later as bounded conversational context.
        """
        if not isinstance(role, str) or not role:
            raise ValueError("role must be a non-empty string")
        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string")

        text = self.truncate_to_tokens(text, self.max_event_tokens)
        event = types.Content(role=role, parts=[types.Part(text=text)])
        self.history.append(event)
        self.trim_in_place()
        self.persist_event({"role": role, "text": text})
        self.compact_file_if_needed()

    def as_dialog_text(self) -> str:
        """Render current in-memory transcript as simple dialogue text."""
        lines: list[str] = []
        for msg in self.history:
            role = getattr(msg, "role", None)
            text = self.content_text(msg)
            if isinstance(role, str) and role and text:
                lines.append(f"{role}: {text}")
        return "\n".join(lines).strip()

    def persist_event(self, event: dict[str, object]) -> None:
        """Persist one transcript event to JSONL.

        Raises:
            RuntimeError: If the transcript cannot be written.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(payload)
        except OSError as e:
            raise RuntimeError(
                f"Failed to persist transcript event to {self.path}: {e}. "
                f"Fix by ensuring the directory is writable and disk has space."
            ) from e

    def clear(self) -> None:
        """Clear history and delete transcript file if present."""
        self.history.clear()
        if self.path.exists():
            self.path.unlink()

    def build_prompt(self) -> str:
        """Build a bounded transcript prompt for a model turn."""
        transcript = "\n".join(self.content_text(item) for item in self.history).strip()
        transcript = self.truncate_to_tokens(transcript, self.max_tokens)
        return (
            "You are participating in a multi-agent brainstorm.\n"
            "Continue the conversation based on the transcript below.\n\n"
            f"{transcript}\n"
        )

    def compact_file_if_needed(self) -> None:
        """Rewrite transcript file to trimmed in-memory history when large."""
        if not self.path.exists():
            return
        if self.path.stat().st_size <= self.compact_after_bytes:
            return
        self.rewrite_file_from_history()

    def rewrite_file_from_history(self) -> None:
        """Rewrite JSONL transcript from current in-memory history."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for msg in self.history:
                role = getattr(msg, "role", None)
                text = self.content_text(msg)
                if isinstance(role, str) and role and text:
                    f.write(json.dumps({"role": role, "text": text}, ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def trim_in_place(self) -> None:
        """Trim history to item and token budgets (keeping the newest messages)."""
        if len(self.history) > self.max_items:
            self.history = self.history[-self.max_items :]

        budget = self.max_tokens
        kept_reversed: list[types.Content] = []
        used = 0
        minimum_keep = min(8, len(self.history))

        for msg in reversed(self.history):
            text = self.content_text(msg)
            cost = self.token_count(text) + 8
            if kept_reversed and used + cost > budget and len(kept_reversed) >= minimum_keep:
                break
            kept_reversed.append(msg)
            used += cost

        self.history = list(reversed(kept_reversed))

    def content_text(self, msg: types.Content) -> str:
        """Extract joined text parts from a Content message."""
        parts = getattr(msg, "parts", None) or []
        out: list[str] = []
        for part in parts:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                out.append(text)
        return "\n".join(out).strip()

    def token_count(self, text: str) -> int:
        """Count tokens for text using the configured tokenizer."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        # Different providers tokenize differently. We deliberately over-estimate so
        # our transcript budget stays safe across Anthropic/OpenAI/Gemini.
        encoding_tokens = len(self.encoding.encode(text))
        char_tokens = max(1, len(text) // 3)
        return max(encoding_tokens, char_tokens)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to a maximum token count by trimming the middle."""
        if self.token_count(text) <= max_tokens:
            return text

        # Use a conservative character cap to avoid provider tokenization surprises.
        # Roughly: >= 3 chars/token for safety.
        max_chars = max(256, max_tokens * 3)
        if len(text) <= max_chars:
            tokens = self.encoding.encode(text)
            keep_start = max_tokens // 2
            keep_end = max_tokens - keep_start
            start = self.encoding.decode(tokens[:keep_start])
            end = self.encoding.decode(tokens[-keep_end:])
            return start + "\n\n[... truncated ...]\n\n" + end

        marker = "\n\n[... truncated ...]\n\n"
        keep_start_chars = max_chars // 2
        keep_end_chars = max_chars - keep_start_chars - len(marker)
        return text[:keep_start_chars] + marker + text[-keep_end_chars:]

