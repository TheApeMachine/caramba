"use client";

import katex from "katex";
import type React from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
	type AstNode,
	buildPreset,
	cloneAst,
	findNodeById,
	innerPlain,
	MathAstBuilder,
	matchPresetFromLatex,
	normalizeLatexComparable,
	type PresetKey,
	replaceNodeById,
	rootToInteractiveKatex,
} from "#/components/latex/math-subselection/ast";
import { EquationStructureChrome } from "#/components/latex/math-subselection/equation-structure-chrome";
import { STRUCTURE_TRANSFORMS } from "#/components/latex/math-subselection/transforms";

/*
EquationStructureEditor ports the math subselection demo into the paper editor:
clickable AST in KaTeX, transforms, presets, undo, and terminal rename — synced as LaTeX.
*/
export function EquationStructureEditor({
	latex,
	displayMode,
	onLatexChange,
	disabled,
	equationSurfaceRef,
	surfaceInteractive,
	onStructureDetachedChange,
}: {
	latex: string;
	displayMode: boolean;
	onLatexChange: (next: string) => void;
	disabled?: boolean;
	equationSurfaceRef: React.RefObject<HTMLDivElement | null>;
	surfaceInteractive: boolean;
	onStructureDetachedChange?: (detached: boolean) => void;
}) {
	const builderReference = useRef(new MathAstBuilder());
	const miniReference = useRef<HTMLSpanElement>(null);
	const lastEmittedReference = useRef<string | null>(null);
	const seededEmptyReference = useRef(false);

	const [root, setRoot] = useState<AstNode>(() =>
		buildPreset(builderReference.current, "default"),
	);
	const [history, setHistory] = useState<AstNode[]>(() => [
		cloneAst(buildPreset(builderReference.current, "default")),
	]);
	const [selectedId, setSelectedId] = useState<number | null>(null);
	const [structureDetached, setStructureDetached] = useState(false);
	const rootReference = useRef(root);

	rootReference.current = root;

	const commitRoot = useCallback(
		(next: AstNode, recordHistory: boolean) => {
			if (recordHistory) {
				setHistory((previous) => [...previous, cloneAst(next)]);
			}

			const plain = innerPlain(next);
			lastEmittedReference.current = plain;
			onLatexChange(plain);
			setRoot(next);
		},
		[onLatexChange],
	);

	useEffect(() => {
		if (!latex.trim() && !seededEmptyReference.current) {
			seededEmptyReference.current = true;
			const next = buildPreset(builderReference.current, "default");
			const plain = innerPlain(next);
			lastEmittedReference.current = plain;
			onLatexChange(plain);
			setRoot(next);
			setHistory([cloneAst(next)]);
			setStructureDetached(false);

			return;
		}

		const emitted = lastEmittedReference.current;

		if (emitted !== null) {
			if (
				normalizeLatexComparable(emitted) === normalizeLatexComparable(latex)
			) {
				lastEmittedReference.current = null;
			}

			return;
		}

		setRoot((current) => {
			if (
				normalizeLatexComparable(innerPlain(current)) ===
				normalizeLatexComparable(latex)
			) {
				return current;
			}

			if (!latex.trim()) {
				const next = buildPreset(builderReference.current, "default");
				setHistory([cloneAst(next)]);
				setSelectedId(null);
				setStructureDetached(false);
				const plain = innerPlain(next);

				if (
					normalizeLatexComparable(plain) !== normalizeLatexComparable(latex)
				) {
					queueMicrotask(() => {
						lastEmittedReference.current = plain;
						onLatexChange(plain);
					});
				}

				return next;
			}

			const matched = matchPresetFromLatex(latex);

			if (matched) {
				setHistory([cloneAst(matched)]);
				setSelectedId(null);
				setStructureDetached(false);

				return matched;
			}

			setStructureDetached(true);

			return current;
		});
	}, [latex, onLatexChange]);

	useEffect(() => {
		onStructureDetachedChange?.(structureDetached);
	}, [structureDetached, onStructureDetachedChange]);

	useEffect(() => {
		const selected =
			selectedId === null ? null : findNodeById(root, selectedId);
		const element = miniReference.current;

		if (!element || !selected) {
			return;
		}

		try {
			katex.render(innerPlain(selected), element, {
				throwOnError: false,
				displayMode: false,
			});
		} catch {
			element.textContent = innerPlain(selected);
		}
	}, [root, selectedId]);

	useEffect(() => {
		const stageElement = equationSurfaceRef.current;

		if (!stageElement || !surfaceInteractive || structureDetached || disabled) {
			return;
		}

		const katexInput = rootToInteractiveKatex(root);

		try {
			katex.render(katexInput, stageElement, {
				displayMode,
				throwOnError: false,
				trust: true,
				strict: "ignore",
			});
		} catch {
			stageElement.textContent = innerPlain(root);

			return;
		}

		const onBackgroundClick = () => {
			setSelectedId(null);
		};

		stageElement.addEventListener("click", onBackgroundClick);

		const cleanups: Array<() => void> = [];

		stageElement.querySelectorAll(".ast-node").forEach((nodeElement) => {
			const handleNodeClick = (event: Event) => {
				event.stopPropagation();
				const match = nodeElement.className.toString().match(/node-(\d+)/);

				if (!match) {
					return;
				}

				setSelectedId(Number(match[1]));
			};

			nodeElement.addEventListener("click", handleNodeClick);
			cleanups.push(() =>
				nodeElement.removeEventListener("click", handleNodeClick),
			);

			const match = nodeElement.className.toString().match(/node-(\d+)/);

			if (match) {
				nodeElement.classList.toggle(
					"selected",
					Number(match[1]) === selectedId,
				);
			}
		});

		return () => {
			stageElement.removeEventListener("click", onBackgroundClick);
			cleanups.forEach((unsub) => {
				unsub();
			});
		};
	}, [
		root,
		selectedId,
		structureDetached,
		disabled,
		displayMode,
		surfaceInteractive,
		equationSurfaceRef,
	]);

	const applyTransform = useCallback(
		(transformId: string) => {
			const transform = STRUCTURE_TRANSFORMS.find(
				(item) => item.id === transformId,
			);

			if (!transform) {
				return;
			}

			const builder = builderReference.current;
			let nextRoot: AstNode;
			let nextSelected: number | null = selectedId;

			if (selectedId === null) {
				nextRoot = transform.apply(builder, root);
			} else {
				let capturedId: number | null = null;
				nextRoot = replaceNodeById(root, selectedId, (node) => {
					const nextSubtree = transform.apply(builder, node);
					capturedId = nextSubtree._id;

					return nextSubtree;
				});
				nextSelected = capturedId;
			}

			setSelectedId(nextSelected);
			commitRoot(nextRoot, true);
		},
		[commitRoot, root, selectedId],
	);

	const undo = useCallback(() => {
		setHistory((previous) => {
			if (previous.length <= 1) {
				return previous;
			}

			const nextHistory = previous.slice(0, -1);
			const snapshot = cloneAst(nextHistory[nextHistory.length - 1]);
			const plain = innerPlain(snapshot);
			lastEmittedReference.current = plain;
			onLatexChange(plain);
			setRoot(snapshot);
			setSelectedId(null);

			return nextHistory;
		});
	}, [onLatexChange]);

	const loadPresetKey = useCallback(
		(key: PresetKey) => {
			const next = buildPreset(builderReference.current, key);
			setHistory([cloneAst(next)]);
			setSelectedId(null);
			setStructureDetached(false);
			commitRoot(next, false);
		},
		[commitRoot],
	);

	const onRenameAtomic = useCallback(
		(value: string) => {
			if (value === "" || selectedId === null) {
				return;
			}

			setRoot((current) => {
				const updated = replaceNodeById(current, selectedId, (node) => {
					if (node.type === "var") {
						return { ...node, name: value };
					}

					if (node.type === "num") {
						const parsed = parseFloat(value);

						if (Number.isNaN(parsed)) {
							return node;
						}

						return { ...node, value: parsed };
					}

					return node;
				});
				const plain = innerPlain(updated);
				lastEmittedReference.current = plain;
				onLatexChange(plain);

				return updated;
			});
		},
		[onLatexChange, selectedId],
	);

	const onRenameCommit = useCallback(() => {
		setHistory((previous) => [...previous, cloneAst(rootReference.current)]);
	}, []);

	const selectedNode =
		selectedId === null ? null : findNodeById(root, selectedId);

	return (
		<EquationStructureChrome
			disabled={disabled}
			historyLength={history.length}
			miniReference={miniReference}
			onApplyTransform={applyTransform}
			onClearSelection={() => {
				setSelectedId(null);
			}}
			onLoadPreset={loadPresetKey}
			onRenameAtomic={onRenameAtomic}
			onRenameCommit={onRenameCommit}
			onUndo={undo}
			root={root}
			selectedId={selectedId}
			selectedNode={selectedNode}
			structureDetached={structureDetached}
		/>
	);
}
