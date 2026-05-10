"use client";

import type React from "react";
import { useEffect, useRef } from "react";
import { usePaperEditor } from "#/components/latex/context";
import type { HeadingLevel, PaperBlock } from "#/components/latex/model/types";
import { Flex } from "#/components/ui/flex";
import { Textarea } from "#/components/ui/textarea";

const headingClass: Record<HeadingLevel, string> = {
	1: "text-2xl font-semibold tracking-tight sm:text-3xl",
	2: "text-xl font-semibold tracking-tight sm:text-2xl",
	3: "text-lg font-semibold sm:text-xl",
};

const kindSelectValue = (block: PaperBlock): string => {
	if (block.type === "heading") {
		return `h${block.level}`;
	}
	return "p";
};

export const BlockRow = ({ block }: { block: PaperBlock }) => {
	const {
		blocks,
		updateText,
		insertParagraphAfter,
		removeBlockAndFocusPrevious,
		setBlockKind,
		setFocusedBlockId,
		registerBlockAnchor,
	} = usePaperEditor();

	const shellRef = useRef<HTMLDivElement>(null);

	useEffect(() => {
		const el = shellRef.current;
		registerBlockAnchor(block.id, el);
		return () => registerBlockAnchor(block.id, null);
	}, [block.id, registerBlockAnchor]);

	const onChangeKind: React.ChangeEventHandler<HTMLSelectElement> = (e) => {
		const v = e.target.value;
		if (v === "p") {
			setBlockKind(block.id, "paragraph");
			return;
		}
		const level = Number(v.replace("h", "")) as HeadingLevel;
		setBlockKind(block.id, "heading", level);
	};

	const onKeyDown: React.KeyboardEventHandler<HTMLTextAreaElement> = (e) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			if (block.type === "heading") {
				insertParagraphAfter(block.id, "");
				return;
			}
			const el = e.currentTarget;
			const start = el.selectionStart;
			const end = el.selectionEnd;
			const text = block.text;
			const before = text.slice(0, start);
			const after = text.slice(end);
			updateText(block.id, before);
			insertParagraphAfter(block.id, after);
			return;
		}

		if (e.key === "Backspace" && block.text === "" && blocks.length > 1) {
			e.preventDefault();
			removeBlockAndFocusPrevious(block.id);
		}
	};

	return (
		<Flex.Row
			className="group gap-2 rounded-lg py-1 pr-1 pl-1 hover:bg-muted/40"
			data-block-id={block.id}
			ref={shellRef}
		>
			<label className="sr-only" htmlFor={`block-kind-${block.id}`}>
				Block type
			</label>
			<select
				className="mt-1 h-8 w-22 shrink-0 cursor-pointer rounded-md border border-border bg-background px-1.5 text-muted-foreground text-xs outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
				id={`block-kind-${block.id}`}
				value={kindSelectValue(block)}
				onChange={onChangeKind}
			>
				<option value="h1">Heading 1</option>
				<option value="h2">Heading 2</option>
				<option value="h3">Heading 3</option>
				<option value="p">Paragraph</option>
			</select>
			<Textarea
				unstyled
				className={
					block.type === "heading"
						? `${headingClass[block.level]} min-h-10 border-0 bg-transparent py-1.5 shadow-none ring-0 focus-visible:ring-0`
						: "min-h-10 border-0 bg-transparent py-2 text-base leading-relaxed shadow-none ring-0 focus-visible:ring-0 sm:text-[17px]"
				}
				placeholder={
					block.type === "heading"
						? "Section title…"
						: "Write here. Enter: new block. Shift+Enter: line break."
				}
				value={block.text}
				onFocus={() => setFocusedBlockId(block.id)}
				onKeyDown={onKeyDown}
				onChange={(e) => updateText(block.id, e.target.value)}
			/>
		</Flex.Row>
	);
};
