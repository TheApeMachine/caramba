"use client";

import katex from "katex";
import "katex/dist/katex.min.css";
import type React from "react";
import { useEffect, useRef, useState } from "react";
import { usePaperEditor } from "#/components/latex/context";
import type { PaperEquationBlock } from "#/components/latex/model/types";
import { Textarea } from "#/components/ui/textarea";

/*
EquationBlock renders a KaTeX preview above a LaTeX source editor.
Click/Enter the rendered equation to edit; blur or Escape to close the editor.
Enter on an empty equation inserts a paragraph after it.
*/
export function EquationBlock({ block, onFocus }: { block: PaperEquationBlock; onFocus?: () => void }) {
	const { updateLatex, insertParagraphAfter, removeBlockAndFocusPrevious, blocks } =
		usePaperEditor();
	const [editing, setEditing] = useState(block.latex === "");
	const textareaRef = useRef<HTMLTextAreaElement>(null);
	const previewRef = useRef<HTMLButtonElement>(null);

	useEffect(() => {
		if (editing) textareaRef.current?.focus();
	}, [editing]);

	useEffect(() => {
		const el = previewRef.current;
		if (!el) return;
		if (!block.latex.trim()) {
			el.innerHTML = '<span class="text-muted-foreground text-sm italic">Click to enter equation…</span>';
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

	const onPreviewKeyDown = (e: React.KeyboardEvent) => {
		if (e.key === "Enter" || e.key === " ") {
			e.preventDefault();
			openEditor();
		}
	};

	const onTextareaKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
		if (e.key === "Escape") { setEditing(false); return; }
		if (e.key === "Enter" && !e.shiftKey && block.latex.trim() === "") {
			e.preventDefault();
			insertParagraphAfter(block.id);
			return;
		}
		if (e.key === "Backspace" && block.latex === "" && blocks.length > 1) {
			e.preventDefault();
			removeBlockAndFocusPrevious(block.id);
		}
	};

	return (
		<div className="flex flex-col gap-1.5">
			<button
				type="button"
				ref={previewRef}
				className="min-h-10 w-full cursor-text rounded-md px-2 py-2 text-center hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
				onClick={openEditor}
				onKeyDown={onPreviewKeyDown}
				onFocus={onFocus}
			/>
			{editing && (
				<Textarea
					ref={textareaRef}
					unstyled
					className="min-h-8 rounded-md border-0 bg-muted/30 px-2 py-1.5 font-mono text-sm shadow-none ring-0 focus-visible:ring-1 focus-visible:ring-ring"
					placeholder="\frac{a}{b} = c"
					value={block.latex}
					onChange={(e) => updateLatex(block.id, e.target.value)}
					onKeyDown={onTextareaKeyDown}
					onBlur={() => setEditing(false)}
				/>
			)}
		</div>
	);
}
