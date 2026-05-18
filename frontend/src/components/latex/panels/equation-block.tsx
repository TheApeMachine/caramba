"use client";

import katex from "katex";
import "katex/dist/katex.min.css";
import "#/components/latex/math-subselection/equation-structure-stage.css";
import type React from "react";
import {
	type FocusEventHandler,
	type KeyboardEventHandler,
	useCallback,
	useEffect,
	useRef,
	useState,
} from "react";
import { usePaperEditor } from "#/components/latex/context";
import { EquationStructureEditor } from "#/components/latex/math-subselection/equation-structure-editor";
import type { PaperEquationBlock } from "#/components/latex/model/types";
import { EditableBlock } from "#/components/latex/panels/editable-block";
import { Button } from "#/components/ui/button";
import { cn } from "#/lib/utils";

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
	const [chromeOpen, setChromeOpen] = useState(block.latex === "");
	const [editing, setEditing] = useState(false);
	const [structureDetached, setStructureDetached] = useState(false);
	const editableRef = useRef<HTMLDivElement>(null);
	const equationSurfaceRef = useRef<HTMLDivElement>(null);
	const equationShellRef = useRef<HTMLDivElement>(null);
	const editingReference = useRef(editing);
	const chromeOpenReference = useRef(chromeOpen);

	editingReference.current = editing;
	chromeOpenReference.current = chromeOpen;

	useEffect(() => {
		if (editing) {
			editableRef.current?.focus();
		}
	}, [editing]);

	const editorOwnsEquationSurface =
		chromeOpen && !editing && !structureDetached;

	useEffect(() => {
		const element = equationSurfaceRef.current;

		if (!element) {
			return;
		}

		if (editorOwnsEquationSurface) {
			return;
		}

		if (!block.latex.trim()) {
			element.innerHTML =
				'<span class="text-muted-foreground text-sm italic">Click to enter equation…</span>';

			return;
		}

		try {
			katex.render(block.latex, element, {
				displayMode: block.display,
				throwOnError: false,
				errorColor: "hsl(var(--destructive))",
			});
		} catch {
			element.textContent = block.latex;
		}
	}, [block.display, block.latex, editorOwnsEquationSurface]);

	const closeEquationChrome = useCallback(() => {
		setChromeOpen(false);
		setEditing(false);
	}, []);

	const openEquationChrome = () => {
		setChromeOpen(true);
	};

	useEffect(() => {
		if (!chromeOpen) {
			return;
		}

		const onPointerDownCapture = (event: PointerEvent) => {
			const shell = equationShellRef.current;
			const target = event.target;

			if (!(target instanceof Node) || !shell) {
				return;
			}

			if (shell.contains(target)) {
				return;
			}

			if (target instanceof Element && target.closest("[data-portal]")) {
				return;
			}

			closeEquationChrome();
		};

		document.addEventListener("pointerdown", onPointerDownCapture, true);

		return () => {
			document.removeEventListener("pointerdown", onPointerDownCapture, true);
		};
	}, [chromeOpen, closeEquationChrome]);

	useEffect(() => {
		if (!chromeOpen) {
			return;
		}

		const onEscape = (event: KeyboardEvent) => {
			if (event.key !== "Escape") {
				return;
			}

			event.preventDefault();

			if (editingReference.current) {
				setEditing(false);

				return;
			}

			setChromeOpen(false);
		};

		window.addEventListener("keydown", onEscape, true);

		return () => {
			window.removeEventListener("keydown", onEscape, true);
		};
	}, [chromeOpen]);

	const onPreviewKeyDown = (event: React.KeyboardEvent) => {
		if (event.key === "Enter" || event.key === " ") {
			event.preventDefault();
			openEquationChrome();
		}
	};

	const onEditorKeyDown: KeyboardEventHandler<HTMLDivElement> = (event) => {
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

	const handleEditorBlur: FocusEventHandler<HTMLDivElement> = (event) => {
		const relatedTarget = event.relatedTarget;

		window.setTimeout(() => {
			window.requestAnimationFrame(() => {
				const shell = equationShellRef.current;
				const active = document.activeElement;

				if (shell?.contains(active)) {
					return;
				}

				if (relatedTarget instanceof Node && shell?.contains(relatedTarget)) {
					return;
				}

				if (active instanceof Element && active.closest("[data-portal]")) {
					return;
				}

				if (!chromeOpenReference.current) {
					return;
				}

				setEditing(false);
			});
		});
	};

	const structurePanelHidden = !chromeOpen || editing;

	return (
		<div className="flex flex-col gap-3" ref={equationShellRef}>
			<article
				aria-label={chromeOpen ? "Equation" : "Edit equation"}
				className={cn(
					"min-h-10 w-full rounded-md px-2 py-2 text-center",
					editorOwnsEquationSurface && "math-subselection-stage",
					!chromeOpen &&
						"cursor-text hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
				)}
				onClick={chromeOpen ? undefined : openEquationChrome}
				onFocus={onFocus}
				onKeyDown={chromeOpen ? undefined : onPreviewKeyDown}
				ref={equationSurfaceRef}
				role={chromeOpen ? undefined : "button"}
				tabIndex={chromeOpen ? -1 : 0}
			/>

			{chromeOpen ? (
				<div className="flex flex-wrap items-center justify-end gap-2">
					{!editing ? (
						<Button
							className="h-8 font-mono text-xs"
							onClick={() => {
								setEditing(true);
							}}
							type="button"
							variant="ghost"
						>
							LaTeX source
						</Button>
					) : null}
					<Button
						className="h-8"
						onClick={closeEquationChrome}
						type="button"
						variant="secondary"
					>
						Done
					</Button>
				</div>
			) : null}

			{editing ? (
				<EditableBlock
					className="min-h-8 rounded-md bg-muted/30 px-2 py-1.5 font-mono text-sm focus-visible:ring-1 focus-visible:ring-ring"
					onBlur={handleEditorBlur}
					onChange={(value) => updateLatex(block.id, value)}
					onKeyDown={onEditorKeyDown}
					placeholder="\frac{a}{b} = c"
					ref={editableRef}
					value={block.latex}
				/>
			) : null}

			<div
				aria-hidden={structurePanelHidden}
				className={cn(structurePanelHidden && "hidden")}
			>
				<EquationStructureEditor
					displayMode={block.display}
					equationSurfaceRef={equationSurfaceRef}
					latex={block.latex}
					onLatexChange={(value) => updateLatex(block.id, value)}
					onStructureDetachedChange={setStructureDetached}
					surfaceInteractive={editorOwnsEquationSurface}
				/>
			</div>
		</div>
	);
}
