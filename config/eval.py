"""
eval provides config models for behavioral evaluation suites.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from caramba.config import PositiveInt, Probability


class TiktokenTokenizerConfig(BaseModel):
    """
    TiktokenTokenizerConfig configures a tiktoken tokenizer.
    """
    type: Literal["tiktoken"] = "tiktoken"
    encoding: str


class LlamaTokenizerConfig(BaseModel):
    """
    LlamaTokenizerConfig configures a Llama/HuggingFace tokenizer.
    """
    type: Literal["llama"] = "llama"
    model_id: str = "meta-llama/Llama-3.2-1B"


class CodeBpeTokenizerConfig(BaseModel):
    """
    CodeBpeTokenizerConfig configures a deterministic CodeBPE tokenizer loaded
    from a local HuggingFace `tokenizers` JSON file.
    """

    type: Literal["code_bpe"] = "code_bpe"
    tokenizer_file: str


# TokenizerConfig is a union of supported tokenizer backends.
# Using Annotated with discriminator allows extension to support
# multiple tokenizer backends without breaking existing configs.
TokenizerConfig = Annotated[
    TiktokenTokenizerConfig | LlamaTokenizerConfig | CodeBpeTokenizerConfig,
    Field(discriminator="type"),
]


class EvalCase(BaseModel):
    """
    EvalCase defines a single behavior eval case with a ground truth.
    """
    id: str
    prompt: str
    kind: Literal["choice_logprob", "int_greedy", "float_greedy", "exact_match_greedy"]

    choices: list[str] | None = None
    answer: str | int | None = None
    # For exact_match_greedy, control how we judge "correct".
    # - exact: output (stripped) must equal answer (stripped)
    # - first_line: compare only the first line of output (stripped)
    # - prefix: output (stripped) must start with answer (stripped)
    match: Literal["exact", "first_line", "prefix"] = "exact"

    @model_validator(mode="after")
    def _validate_case(self) -> "EvalCase":
        """
        _validate_case validates case invariants.
        """
        if not self.id:
            raise ValueError("case.id must be non-empty")
        if not self.prompt:
            raise ValueError("case.prompt must be non-empty")

        match self.kind:
            case "choice_logprob":
                if not self.choices:
                    raise ValueError("choice_logprob requires non-empty choices")
                if not isinstance(self.answer, str) or not self.answer:
                    raise ValueError("choice_logprob requires a string answer")
                if str(self.answer) not in set(self.choices):
                    raise ValueError(
                        "choice_logprob answer must be one of choices"
                    )
            case "int_greedy":
                if not isinstance(self.answer, int):
                    raise ValueError("int_greedy requires an integer answer")
            case "float_greedy":
                if not isinstance(self.answer, (int, float)):
                    raise ValueError("float_greedy requires a numeric answer")
            case "exact_match_greedy":
                if not isinstance(self.answer, str) or not str(self.answer).strip():
                    raise ValueError("exact_match_greedy requires a non-empty string answer")
                if str(self.match) not in {"exact", "first_line", "prefix"}:
                    raise ValueError("exact_match_greedy.match must be one of: exact, first_line, prefix")
            case _:
                raise ValueError(f"Unsupported eval case kind: {self.kind!r}")

        return self


class EvalThresholds(BaseModel):
    """
    EvalThresholds defines acceptance thresholds for eval results.
    """
    min_student_accuracy: Probability
    max_accuracy_drop: Probability


class EvalVerifyConfig(BaseModel):
    """
    EvalVerifyConfig runs a small behavioral suite against teacher and student.
    """
    type: Literal["eval"] = "eval"
    tokenizer: TokenizerConfig
    max_new_tokens: PositiveInt
    context_window: PositiveInt | None = None
    cases: list[EvalCase] = Field(min_length=1)
    thresholds: EvalThresholds


