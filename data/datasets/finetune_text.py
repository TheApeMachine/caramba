"""Text dataset helpers for finetuning."""
from __future__ import annotations

from typing import Any, Iterable, cast

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    load_dataset,
    concatenate_datasets,
    interleave_datasets,
)

from console import logger


def _flatten_message_tree(message: dict, conversation: list[str] | None = None) -> list[str]:
    """Flatten a message tree into conversation strings."""
    if conversation is None:
        conversation = []

    role = message.get("role", "")
    text = message.get("text", "").strip()
    if text:
        conversation.append(f"{'User' if role == 'prompter' else 'Assistant'}: {text}")

    replies = message.get("replies", [])
    if not replies:
        return ["\n\n".join(conversation)]

    all_conversations = []
    for reply in replies:
        branch_conversation = conversation.copy()
        branch_conversations = _flatten_message_tree(reply, branch_conversation)
        all_conversations.extend(branch_conversations)

    return all_conversations


def _tokenize_examples(
    *,
    examples: dict[str, list[Any]],
    tokenizer: Any,
    max_seq_length: int,
    dummy_token_start: int | None = None,
    append_eos: bool = False,
) -> dict[str, list[int] | list[list[int]]]:
    texts: list[str] = []
    # Number of tokens to mask at the start of each example so loss is computed
    # only on the assistant completion (instruction tuning).
    prompt_token_lens: list[int] = []

    if "instruction" in examples:
        for ins, inp, out in zip(
            examples.get("instruction", []),
            examples.get("input", []),
            examples.get("output", []),
        ):
            if inp:
                prompt = f"User: {ins}\n\n{inp}\n\nAssistant: "
            else:
                prompt = f"User: {ins}\n\nAssistant: "
            text = prompt + out
            if append_eos and getattr(tokenizer, "eos_token", None):
                text = text + str(tokenizer.eos_token)
            texts.append(text)
            # Mask prompt (including "Assistant:" anchor) so we only learn the answer tokens.
            try:
                prompt_ids = tokenizer(
                    prompt, truncation=True, max_length=max_seq_length, padding=False
                )["input_ids"]
                prompt_token_lens.append(int(len(prompt_ids)))
            except Exception:
                prompt_token_lens.append(0)

    elif "prompt" in examples:
        for prompt_obj in examples.get("prompt", []):
            if isinstance(prompt_obj, dict):
                conversations = _flatten_message_tree(prompt_obj)
                for conversation in conversations:
                    text = conversation
                    if append_eos and getattr(tokenizer, "eos_token", None):
                        text = text + str(tokenizer.eos_token)
                    texts.append(text)
                    # No reliable prompt/completion boundary here; train on all non-pad tokens.
                    prompt_token_lens.append(0)

    elif "text" in examples and "role" in examples:
        for text, role in zip(examples.get("text", []), examples.get("role", [])):
            if text and role:
                formatted = f"{'User' if role == 'prompter' else 'Assistant'}: {text}"
                text = formatted
                if append_eos and getattr(tokenizer, "eos_token", None):
                    text = text + str(tokenizer.eos_token)
                texts.append(text)
                # If this is a user-only record, mask everything. If assistant-only, keep all.
                prompt_token_lens.append(max_seq_length if role == "prompter" else 0)

    if not texts:
        for key in examples.keys():
            if "text" in key.lower() and isinstance(examples[key], list):
                for t in examples[key]:
                    if not t:
                        continue
                    text = str(t)
                    if append_eos and getattr(tokenizer, "eos_token", None):
                        text = text + str(tokenizer.eos_token)
                    texts.append(text)
                    prompt_token_lens.append(0)
                break

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )
    # IMPORTANT: tokenizer returns a list-of-lists for batched calls.
    # A shallow `.copy()` would share inner lists and mutations would corrupt
    # `input_ids` (e.g. inserting -100), which then crashes embedding lookups.
    input_ids = tokenized["input_ids"]
    labels = [list(seq) for seq in input_ids]

    # Ensure we have one prompt length per tokenized example.
    if len(prompt_token_lens) != len(labels):
        prompt_token_lens = [0 for _ in range(len(labels))]

    # Mask out:
    # - prompt tokens (instruction tuning)
    # - padding tokens (loss should ignore padding)
    attn = tokenized.get("attention_mask")
    for i, seq in enumerate(labels):
        # Prompt mask (clamp to sequence length).
        p = int(prompt_token_lens[i])
        if p > 0:
            for j in range(min(p, len(seq))):
                seq[j] = -100
        # Padding mask via attention_mask if present.
        if attn is not None:
            am = attn[i]
            for j, m in enumerate(am):
                if int(m) == 0:
                    seq[j] = -100
    if dummy_token_start is not None:
        labels = [
            [-100 if token_id >= dummy_token_start else token_id for token_id in seq]
            for seq in labels
        ]
    tokenized["labels"] = labels
    return tokenized


def build_finetune_text_datasets(
    *,
    tokenizer: Any,
    max_seq_length: int,
    seed: int = 42,
    dataset_ids: Iterable[str],
    dataset_probs: list[float] | None = None,
    streaming: bool = False,
    streaming_shuffle_buffer: int = 10_000,
    dummy_token_start: int | None = None,
    append_eos: bool = False,
) -> tuple[Dataset | IterableDataset, None]:
    """Load and tokenize text finetuning datasets.

    Returns:
        Tuple of (train_dataset, None). Eval dataset should be provided separately.
    """
    dataset_ids = list(dataset_ids)
    if not dataset_ids:
        raise ValueError("dataset_ids must be provided for finetuning")

    streaming_datasets: list[IterableDataset] = []
    static_datasets: list[Dataset] = []

    for dataset_id in dataset_ids:
        # Allow "repo/dataset::config_name" syntax (HuggingFace dataset config/subset)
        original_dataset_id = dataset_id
        dataset_name: str | None = None
        if "::" in dataset_id:
            dataset_id, dataset_name = dataset_id.split("::", 1)

        logger.info(
            f"Loading {dataset_id} (config={dataset_name}, streaming={streaming}, append_eos={append_eos})"
        )

        try:
            if streaming:
                dataset = cast(
                    IterableDataset,
                    load_dataset(dataset_id, name=dataset_name, split="train", streaming=True),
                )
                remove_columns = list(getattr(dataset, "features", {}).keys())
                tokenized = dataset.map(
                    lambda batch: _tokenize_examples(
                        examples=batch,
                        tokenizer=tokenizer,
                        max_seq_length=max_seq_length,
                        dummy_token_start=dummy_token_start,
                        append_eos=append_eos,
                    ),
                    batched=True,
                    remove_columns=remove_columns if remove_columns else None,
                )
                streaming_datasets.append(tokenized)
                logger.info(f"Loaded {original_dataset_id} (streaming)")
            else:
                dataset = cast(DatasetDict, load_dataset(dataset_id, name=dataset_name))
                train_split = cast(Dataset, dataset["train"])
                tokenized = dataset.map(
                    lambda batch: _tokenize_examples(
                        examples=batch,
                        tokenizer=tokenizer,
                        max_seq_length=max_seq_length,
                        dummy_token_start=dummy_token_start,
                        append_eos=append_eos,
                    ),
                    batched=True,
                    remove_columns=train_split.column_names,
                )
                tokenized_train = cast(Dataset, tokenized["train"])
                static_datasets.append(tokenized_train)
                logger.info(f"Loaded {original_dataset_id}: {len(tokenized_train)} examples")
        except Exception as e:
            logger.warning(f"Failed to load {original_dataset_id}: {e}")

    if streaming:
        if not streaming_datasets:
            raise ValueError("No datasets were successfully loaded")
        all_datasets = streaming_datasets
    else:
        if not static_datasets:
            raise ValueError("No datasets were successfully loaded")

    if dataset_probs is not None:
        probs = list(dataset_probs)
        expected_len = len(streaming_datasets) if streaming else len(static_datasets)
        if len(probs) != expected_len:
            raise ValueError("dataset_probs must match dataset_ids length")
        total = float(sum(probs))
        if total <= 0:
            raise ValueError("dataset_probs must sum to > 0")
        probs = [float(p) / total for p in probs]
        if streaming:
            tokenized = interleave_datasets(
                streaming_datasets,
                probabilities=probs,
                seed=seed,
                stopping_strategy="all_exhausted",
            )
        else:
            tokenized = interleave_datasets(
                static_datasets,
                probabilities=probs,
                seed=seed,
                stopping_strategy="all_exhausted",
            )
    else:
        if streaming:
            tokenized = interleave_datasets(streaming_datasets, seed=seed)
        else:
            tokenized = concatenate_datasets(static_datasets)

    if streaming:
        tokenized_iterable = cast(IterableDataset, tokenized)
        if streaming_shuffle_buffer > 0:
            tokenized_iterable = tokenized_iterable.shuffle(
                seed=seed, buffer_size=streaming_shuffle_buffer
            )
        logger.info("Prepared streaming dataset (no fixed length).")
        return tokenized_iterable, None

    tokenized_static = cast(Dataset, tokenized)
    logger.info(f"Combined training dataset size: {len(tokenized_static)} examples")
    return tokenized_static, None
