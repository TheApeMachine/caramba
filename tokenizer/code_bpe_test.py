from __future__ import annotations

from pathlib import Path

import pytest

from caramba.tokenizer.code_bpe import CodeBpeTokenizer


class TestCodeBpeTokenizer:
    """Tests for CodeBpeTokenizer."""

    def test_train_and_load_roundtrip(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "src"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "a.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        (data_dir / "b.go").write_text("package main\n\nfunc main() {}\n", encoding="utf-8")

        out_file = tmp_path / "tok.json"
        tok = CodeBpeTokenizer(
            vocab_size=128,
            special_tokens=["<unk>", "<pad>", "<s>", "</s>"],
            file_extensions=[".py", ".go"],
        )
        saved = tok.train(data_dir=str(data_dir), output_file=str(out_file))
        assert saved.exists()

        loaded = tok.load(tokenizer_file=str(saved))
        pad_id = loaded.token_to_id("<pad>")
        assert isinstance(pad_id, int)
        assert pad_id >= 0

        text = "def foo():\n    return 1\n"
        ids = loaded.encode(text).ids
        assert isinstance(ids, list)
        assert len(ids) > 1
        assert all(isinstance(i, int) for i in ids)
        assert all(i >= 0 for i in ids)

        # Deterministic encoding for the same input text.
        ids2 = loaded.encode(text).ids
        assert ids == ids2

        # Special token handling: <pad> should not appear when we're not padding.
        assert pad_id not in ids

