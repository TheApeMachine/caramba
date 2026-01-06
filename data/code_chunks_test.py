from __future__ import annotations

from pathlib import Path

from caramba.data.code_chunks import CodeChunksDataset
from caramba.tokenizer.code_bpe import CodeBpeTokenizer


class TestCodeChunksDataset:
    """Tests for CodeChunksDataset."""

    def test_yields_padded_target_and_prompt(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "src"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "a.py").write_text("x = 1\n" * 200, encoding="utf-8")

        tokenizer_file = tmp_path / "tok.json"
        CodeBpeTokenizer(
            vocab_size=128,
            special_tokens=["<unk>", "<pad>", "<s>", "</s>"],
            file_extensions=[".py"],
        ).train(data_dir=str(data_dir), output_file=str(tokenizer_file))

        ds = CodeChunksDataset(
            data_dir=str(data_dir),
            tokenizer_file=str(tokenizer_file),
            seq_len=32,
            stride=16,
            extensions=[".py"],
            max_files=None,
            cache_size=4,
        ).build()

        item = ds[0]
        assert "target_ids" in item and "prompt_ids" in item
        assert tuple(item["target_ids"].shape) == (32,)
        assert tuple(item["prompt_ids"].shape) == (32,)

