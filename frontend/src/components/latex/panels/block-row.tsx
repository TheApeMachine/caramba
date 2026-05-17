"use client";

import { GripVerticalIcon, PlusIcon, TrashIcon } from "lucide-react";
import type React from "react";
import { useEffect, useRef, useState } from "react";
import { usePaperEditor } from "#/components/latex/context";
import {
	type BlockKindDescriptor,
	matchMarkdownShortcut,
} from "#/components/latex/model/block-catalog";
import type {
	HeadingLevel,
	PaperBlock,
	PaperBlockKind,
} from "#/components/latex/model/types";
import { BlockKindMenu } from "#/components/latex/panels/block-kind-menu";
import {
	caretOffset,
	EditableBlock,
} from "#/components/latex/panels/editable-block";
import { EquationBlock } from "#/components/latex/panels/equation-block";
import { FormattingToolbar } from "#/components/latex/panels/formatting-toolbar";
import { Button } from "#/components/ui/button";
import { useDragDrop } from "#/components/ui/drag-drop";
import { Flex } from "#/components/ui/flex";

type BlockDragPayload = { kind: "paper-block"; blockId: string };

const headingClass: Record<HeadingLevel, string> = {
	1: "text-2xl font-semibold tracking-tight sm:text-3xl",
	2: "text-xl font-semibold tracking-tight sm:text-2xl",
	3: "text-lg font-semibold sm:text-xl",
};

type TextualBlock = Exclude<PaperBlock, { type: "equation" }>;

function placeholderFor(block: TextualBlock): string {
	if (block.type === "heading") {
		return "Section title…";
	}

	if (block.type === "list") {
		return "One item per line";
	}

	return "Write here. Press / for blocks, # for headings, $$ for math.";
}

function editableClass(block: TextualBlock): string {
	if (block.type === "heading") {
		return `${headingClass[block.level]} min-h-10 py-1.5`;
	}

	if (block.type === "list") {
		return "min-h-10 py-2 font-mono text-sm leading-relaxed";
	}

	return "min-h-10 py-2 text-base leading-relaxed sm:text-[17px]";
}

function applyDescriptorToBlock(
	descriptor: BlockKindDescriptor,
	resetText: () => void,
	setBlockKind: (
		id: string,
		kind: PaperBlockKind,
		options?: { level?: HeadingLevel; ordered?: boolean },
	) => void,
	blockId: string,
): void {
	const sample = descriptor.build();
	resetText();

	if (sample.type === "heading") {
		setBlockKind(blockId, "heading", { level: sample.level });
		return;
	}

	if (sample.type === "list") {
		setBlockKind(blockId, "list", { ordered: sample.ordered });
		return;
	}

	setBlockKind(blockId, sample.type);
}

function readDropPosition(
	event: React.DragEvent<HTMLDivElement>,
): "above" | "below" {
	const rect = event.currentTarget.getBoundingClientRect();
	const midpoint = rect.top + rect.height / 2;
	return event.clientY < midpoint ? "above" : "below";
}

export const BlockRow = ({ block }: { block: PaperBlock }) => {
	const {
		blocks,
		updateText,
		insertParagraphAfter,
		insertBlockAfter,
		removeBlockAndFocusPrevious,
		reorderBlock,
		setBlockKind,
		setFocusedBlockId,
		registerBlockAnchor,
	} = usePaperEditor();

	const dnd = useDragDrop<BlockDragPayload>();
	const shellRef = useRef<HTMLDivElement>(null);
	const editableRef = useRef<HTMLDivElement>(null);
	const [slashOpen, setSlashOpen] = useState(false);
	const [dragArmed, setDragArmed] = useState(false);
	const [dropEdge, setDropEdge] = useState<"above" | "below" | null>(null);
	const [isFocused, setIsFocused] = useState(false);

	const handleFocus = () => {
		setIsFocused(true);
		setFocusedBlockId(block.id);
	};

	const handleBlur: React.FocusEventHandler<HTMLDivElement> = (event) => {
		const next = event.relatedTarget as HTMLElement | null;

		if (next && shellRef.current?.contains(next)) {
			return;
		}

		setIsFocused(false);
	};

	useEffect(() => {
		const el = shellRef.current;
		registerBlockAnchor(block.id, el);
		return () => registerBlockAnchor(block.id, null);
	}, [block.id, registerBlockAnchor]);

	const handleInsertBelow = (descriptor: BlockKindDescriptor) => {
		insertBlockAfter(block.id, descriptor.build());
	};

	const handleSlashSelect = (descriptor: BlockKindDescriptor) => {
		setSlashOpen(false);
		applyDescriptorToBlock(
			descriptor,
			() => updateText(block.id, ""),
			setBlockKind,
			block.id,
		);
	};

	const onDragStart: React.DragEventHandler<HTMLDivElement> = (event) => {
		if (!dragArmed) {
			event.preventDefault();
			return;
		}

		event.dataTransfer.effectAllowed = "move";
		event.dataTransfer.setData("text/plain", block.id);
		dnd.begin({ kind: "paper-block", blockId: block.id });
		event.currentTarget.classList.add("opacity-50");
	};

	const onDragEnd: React.DragEventHandler<HTMLDivElement> = (event) => {
		event.currentTarget.classList.remove("opacity-50");
		setDragArmed(false);
		setDropEdge(null);
	};

	const onDragOver: React.DragEventHandler<HTMLDivElement> = (event) => {
		const payload = dnd.read();

		if (!payload || payload.blockId === block.id) {
			return;
		}

		event.preventDefault();
		event.dataTransfer.dropEffect = "move";
		setDropEdge(readDropPosition(event));
	};

	const onDragLeave: React.DragEventHandler<HTMLDivElement> = () => {
		setDropEdge(null);
	};

	const onDrop: React.DragEventHandler<HTMLDivElement> = (event) => {
		const payload = dnd.read();
		setDropEdge(null);

		if (!payload || payload.blockId === block.id) {
			return;
		}

		event.preventDefault();
		event.stopPropagation();

		const position = readDropPosition(event);
		dnd.end();
		reorderBlock(payload.blockId, block.id, position);
	};

	const gutter = (
		<Flex.Column className="w-8 shrink-0 items-end pt-1.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
			<button
				aria-label="Drag to reorder"
				className="flex size-6 cursor-grab items-center justify-center rounded text-muted-foreground hover:text-foreground active:cursor-grabbing"
				onMouseDown={() => setDragArmed(true)}
				onMouseUp={() => setDragArmed(false)}
				type="button"
			>
				<GripVerticalIcon className="size-3.5" />
			</button>

			<BlockKindMenu
				variant="trigger"
				onSelect={handleInsertBelow}
				trigger={
					<Button
						aria-label="Insert block below"
						className="size-6 text-muted-foreground hover:text-foreground"
						size="icon"
						type="button"
						variant="ghost"
					>
						<PlusIcon className="size-3.5" />
					</Button>
				}
			/>

			<Button
				aria-label="Delete block"
				className="size-6 text-muted-foreground hover:text-destructive"
				disabled={blocks.length <= 1}
				onClick={() => removeBlockAndFocusPrevious(block.id)}
				size="icon"
				type="button"
				variant="ghost"
			>
				<TrashIcon className="size-3.5" />
			</Button>
		</Flex.Column>
	);

	const dropEdgeClass =
		dropEdge === "above"
			? "before:absolute before:inset-x-2 before:top-0 before:h-0.5 before:rounded before:bg-primary"
			: dropEdge === "below"
				? "after:absolute after:inset-x-2 after:bottom-0 after:h-0.5 after:rounded after:bg-primary"
				: "";

	const containerClass = `group relative flex items-start gap-1 rounded-lg py-1 pr-1 pl-1 hover:bg-muted/40 ${dropEdgeClass}`;

	if (block.type === "equation") {
		return (
			// biome-ignore lint/a11y/noStaticElementInteractions: drag-and-drop container, not an interactive control
			<div
				className={containerClass}
				data-block-id={block.id}
				draggable={dragArmed}
				onDragEnd={onDragEnd}
				onDragLeave={onDragLeave}
				onDragOver={onDragOver}
				onDragStart={onDragStart}
				onDrop={onDrop}
				ref={shellRef}
			>
				{gutter}
				<div className="flex-1">
					<EquationBlock
						block={block}
						onFocus={() => setFocusedBlockId(block.id)}
					/>
				</div>
			</div>
		);
	}

	const textual = block;

	const onTextChange = (value: string) => {
		const promoted =
			textual.type === "paragraph" ? matchMarkdownShortcut(value) : null;

		if (promoted) {
			applyDescriptorToBlock(
				promoted,
				() => updateText(block.id, ""),
				setBlockKind,
				block.id,
			);
			return;
		}

		updateText(block.id, value);

		const openSlash = textual.type === "paragraph" && value === "/";
		setSlashOpen(openSlash);
	};

	const onKeyDown: React.KeyboardEventHandler<HTMLDivElement> = (event) => {
		if (slashOpen && event.key === "Escape") {
			event.preventDefault();
			setSlashOpen(false);
			return;
		}

		if (event.key === "Enter" && !event.shiftKey && textual.type !== "list") {
			event.preventDefault();

			if (textual.type === "heading") {
				insertParagraphAfter(block.id, "");
				return;
			}

			const el = event.currentTarget;
			const offset = caretOffset(el);
			const before = textual.text.slice(0, offset);
			const after = textual.text.slice(offset);
			updateText(block.id, before);
			insertParagraphAfter(block.id, after);
			return;
		}

		if (event.key === "Backspace" && textual.text === "" && blocks.length > 1) {
			event.preventDefault();
			removeBlockAndFocusPrevious(block.id);
		}
	};

	return (
		// biome-ignore lint/a11y/noStaticElementInteractions: drag-and-drop container, not an interactive control
		<div
			className={containerClass}
			data-block-id={block.id}
			draggable={dragArmed}
			onDragEnd={onDragEnd}
			onDragLeave={onDragLeave}
			onDragOver={onDragOver}
			onDragStart={onDragStart}
			onDrop={onDrop}
			ref={shellRef}
		>
			{gutter}

			<Flex.Column className="min-w-0 flex-1" gap={1}>
				{textual.type === "list" ? (
					<span className="px-2 text-muted-foreground text-xs uppercase tracking-wide">
						{textual.ordered ? "Numbered list" : "Bullet list"}
					</span>
				) : null}

				<EditableBlock
					className={editableClass(textual)}
					onBlur={handleBlur}
					onChange={onTextChange}
					onFocus={handleFocus}
					onKeyDown={onKeyDown}
					placeholder={placeholderFor(textual)}
					ref={editableRef}
					value={textual.text}
				/>
			</Flex.Column>

			{isFocused ? <FormattingToolbar /> : null}

			<BlockKindMenu
				onOpenChange={setSlashOpen}
				onSelect={handleSlashSelect}
				open={slashOpen}
				variant="anchored"
			/>
		</div>
	);
};
