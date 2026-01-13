from __future__ import annotations

from pathlib import Path

import pytest

from caramba.config.eval import CodeBpeTokenizerConfig, LlamaTokenizerConfig, TiktokenTokenizerConfig
from caramba.data.tokenizers.builder import TokenizerBuilder
from caramba.data.tokenizers.bpe import CodeBpeTokenizer
import caramba.data.tokenizers.huggingface as hf


def test_build_tokenizer_tiktoken() -> None:
    tok = TokenizerBuilder().build_tokenizer(TiktokenTokenizerConfig(encoding="cl100k_base"))
    ids = tok.encode("hello")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids)


def test_build_tokenizer_llama_uses_transformers_autotokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool = False):
            assert add_special_tokens is False
            return [1, 2, 3]

        def decode(self, ids, *, skip_special_tokens: bool = True):
            assert skip_special_tokens is True
            return "ok"

    def _from_pretrained(model_id: str, **kwargs):
        assert model_id
        assert kwargs.get("use_fast") is True
        assert kwargs.get("trust_remote_code") is False
        return _DummyTokenizer()

    monkeypatch.setattr(hf.AutoTokenizer, "from_pretrained", staticmethod(_from_pretrained), raising=True)

    tok = TokenizerBuilder().build_tokenizer(LlamaTokenizerConfig(model_id="meta-llama/Llama-3.2-1B"))
    assert tok.encode("hi") == [1, 2, 3]
    assert tok.decode([1, 2, 3]) == "ok"


def test_build_tokenizer_code_bpe(tmp_path: Path) -> None:
    data_dir = tmp_path / "src"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "a.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (data_dir / "b.go").write_text("package main\n\nfunc main() {}\n", encoding="utf-8")

    out_file = tmp_path / "tok.json"
    trainer = CodeBpeTokenizer(
        vocab_size=128,
        special_tokens=["<unk>", "<pad>", "<s>", "</s>"],
        file_extensions=[".py", ".go"],
    )
    trainer.train(data_dir=str(data_dir), output_file=str(out_file))

    tok = TokenizerBuilder().build_tokenizer(CodeBpeTokenizerConfig(tokenizer_file=str(out_file)))
    ids = tok.encode("def foo():\n    return 1\n")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids)

