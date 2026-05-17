"use client";

import katex from "katex";
import "katex/dist/katex.min.css";
import type React from "react";
import { useEffect, useRef, useState } from "react";
import { usePaperEditor } from "#/components/latex/context";
import type { PaperEquationBlock } from "#/components/latex/model/types";
import { EditableBlock } from "#/components/latex/panels/editable-block";

export function EquationBlock({
	block,
	onFocus,
}: {
	block: PaperEquationBlock;
	onFocus?: () => void;
}) {
	const {
		updateLatex,
		insertParagraphAfter,
		removeBlockAndFocusPrevious,
		blocks,
	} = usePaperEditor();
	const [editing, setEditing] = useState(block.latex === "");
	const editableRef = useRef<HTMLDivElement>(null);
	const previewRef = useRef<HTMLButtonElement>(null);

	useEffect(() => {
		if (editing) {
			editableRef.current?.focus();
		}
	}, [editing]);

	useEffect(() => {
		const el = previewRef.current;

		if (!el) {
			return;
		}

		if (!block.latex.trim()) {
			el.innerHTML =
				'<span class="text-muted-foreground text-sm italic">Click to enter equation…</span>';
			return;
		}

		try {
			katex.render(block.latex, el, {
				displayMode: block.display,
				throwOnError: false,
				errorColor: "hsl(var(--destructive))",
			});
		} catch {
			el.textContent = block.latex;
		}
	}, [block.latex, block.display]);

	const openEditor = () => setEditing(true);

	const onPreviewKeyDown = (event: React.KeyboardEvent) => {
		if (event.key === "Enter" || event.key === " ") {
			event.preventDefault();
			openEditor();
		}
	};

	const onEditorKeyDown: React.KeyboardEventHandler<HTMLDivElement> = (
		event,
	) => {
		if (event.key === "Escape") {
			setEditing(false);
			return;
		}

		if (event.key === "Enter" && !event.shiftKey && block.latex.trim() === "") {
			event.preventDefault();
			insertParagraphAfter(block.id);
			return;
		}

		if (event.key === "Backspace" && block.latex === "" && blocks.length > 1) {
			event.preventDefault();
			removeBlockAndFocusPrevious(block.id);
		}
	};

	return (
		<div className="flex flex-col gap-1.5">
			<button
				className="min-h-10 w-full cursor-text rounded-md px-2 py-2 text-center hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
				onClick={openEditor}
				onFocus={onFocus}
				onKeyDown={onPreviewKeyDown}
				ref={previewRef}
				type="button"
			/>

			{editing ? (
				<EditableBlock
					className="min-h-8 rounded-md bg-muted/30 px-2 py-1.5 font-mono text-sm focus-visible:ring-1 focus-visible:ring-ring"
					onBlur={() => setEditing(false)}
					onChange={(value) => updateLatex(block.id, value)}
					onKeyDown={onEditorKeyDown}
					placeholder="\frac{a}{b} = c"
					ref={editableRef}
					value={block.latex}
				/>
			) : null}
		</div>
	);
}
