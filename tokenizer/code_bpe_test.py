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
        assert loaded.token_to_id("<pad>") is not None
        ids = loaded.encode("def foo():\n    return 1\n").ids
        assert isinstance(ids, list)
        assert len(ids) > 0

