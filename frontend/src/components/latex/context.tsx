"use client";

import type React from "react";
import {
	createContext,
	useCallback,
	useContext,
	useEffect,
	useMemo,
	useReducer,
	useRef,
	useState,
} from "react";
import {
	createInitialPaperBlocks,
	type PaperAction,
	paperReducer,
} from "#/components/latex/model/paper-reducer";
import type { HeadingLevel, PaperBlock, PaperMetadata } from "#/components/latex/model/types";
import { usePaperMetadataForm, type PaperMetadataFormApi } from "#/components/latex/panels/metadata-tab";
import { editorBridge } from "./editor-bridge";

type PaperEditorContextValue = {
	blocks: PaperBlock[];
	dispatch: React.Dispatch<PaperAction>;
	focusedBlockId: string | null;
	setFocusedBlockId: (id: string | null) => void;
	metadataForm: PaperMetadataFormApi;
	updateText: (id: string, text: string) => void;
	updateLatex: (id: string, latex: string) => void;
	insertParagraphAfter: (afterId: string, text?: string) => string;
	insertEquationAfter: (afterId: string, latex?: string) => string;
	insertHeadingAfter: (afterId: string, level: HeadingLevel) => string;
	removeBlockAndFocusPrevious: (id: string) => void;
	setBlockKind: (
		id: string,
		kind: "paragraph" | "heading" | "equation",
		level?: HeadingLevel,
	) => void;
	focusBlock: (id: string) => void;
	registerBlockAnchor: (id: string, el: HTMLElement | null) => void;
	scrollToBlock: (id: string) => void;
};

const PaperEditorContext = createContext<PaperEditorContextValue | null>(null);

export function PaperEditorProvider({
	children,
}: {
	children: React.ReactNode;
}) {
	const [blocks, dispatch] = useReducer(
		paperReducer,
		undefined,
		createInitialPaperBlocks,
	);
	const [focusedBlockId, setFocusedBlockId] = useState<string | null>(null);
	const anchorsRef = useRef(new Map<string, HTMLElement>());
	const metadataForm = usePaperMetadataForm();

	// Stable ref to latest blocks so bridge callbacks are never stale.
	const blocksRef = useRef(blocks);
	blocksRef.current = blocks;

	const registerBlockAnchor = useCallback(
		(id: string, el: HTMLElement | null) => {
			if (el) {
				anchorsRef.current.set(id, el);
			} else {
				anchorsRef.current.delete(id);
			}
		},
		[],
	);

	const focusBlock = useCallback((id: string) => {
		const root = anchorsRef.current.get(id);
		const textarea = root?.querySelector("textarea");
		textarea?.focus();
		root?.scrollIntoView({ behavior: "smooth", block: "center" });
	}, []);

	const scrollToBlock = useCallback(
		(id: string) => {
			focusBlock(id);
		},
		[focusBlock],
	);

	const updateText = useCallback((id: string, text: string) => {
		dispatch({ type: "UPDATE_TEXT", id, text });
	}, []);

	const updateLatex = useCallback((id: string, latex: string) => {
		dispatch({ type: "UPDATE_LATEX", id, latex });
	}, []);

	const insertParagraphAfter = useCallback(
		(afterId: string, text = ""): string => {
			const block: PaperBlock = { id: crypto.randomUUID(), type: "paragraph", text };
			dispatch({ type: "INSERT_AFTER", afterId, block });
			queueMicrotask(() => focusBlock(block.id));
			return block.id;
		},
		[focusBlock],
	);

	const insertEquationAfter = useCallback(
		(afterId: string, latex = ""): string => {
			const block: PaperBlock = { id: crypto.randomUUID(), type: "equation", latex, display: true };
			dispatch({ type: "INSERT_AFTER", afterId, block });
			queueMicrotask(() => focusBlock(block.id));
			return block.id;
		},
		[focusBlock],
	);

	const insertHeadingAfter = useCallback(
		(afterId: string, level: HeadingLevel): string => {
			const block: PaperBlock = { id: crypto.randomUUID(), type: "heading", level, text: "" };
			dispatch({ type: "INSERT_AFTER", afterId, block });
			queueMicrotask(() => focusBlock(block.id));
			return block.id;
		},
		[focusBlock],
	);

	const removeBlockAndFocusPrevious = useCallback(
		(id: string) => {
			const idx = blocksRef.current.findIndex((b) => b.id === id);
			const prevId = idx > 0 ? blocksRef.current[idx - 1]?.id : undefined;
			dispatch({ type: "REMOVE_BLOCK", id });
			if (prevId) queueMicrotask(() => focusBlock(prevId));
		},
		[focusBlock],
	);

	const setBlockKind = useCallback(
		(id: string, kind: "paragraph" | "heading" | "equation", level?: HeadingLevel) => {
			dispatch({ type: "SET_BLOCK_KIND", id, kind, level });
		},
		[],
	);

	// Register / unregister the bridge when this provider mounts / unmounts.
	useEffect(() => {
		editorBridge.register({
			getBlocks: () => blocksRef.current,
			getMetadata: () => metadataForm.store.state.values as PaperMetadata,
			updateText,
			updateLatex,
			insertParagraphAfter,
			insertHeadingAfter,
			insertEquationAfter,
			removeBlock: (id) => dispatch({ type: "REMOVE_BLOCK", id }),
			setBlockKind,
			updateMetadata: (patch) => {
				const current = metadataForm.store.state.values as PaperMetadata;
				for (const [key, val] of Object.entries(patch)) {
					metadataForm.setFieldValue(key as keyof PaperMetadata, val as string);
				}
				void current;
			},
			scrollToBlock,
		});
		return () => editorBridge.unregister();
	}, [
		updateText,
		updateLatex,
		insertParagraphAfter,
		insertHeadingAfter,
		insertEquationAfter,
		setBlockKind,
		scrollToBlock,
		metadataForm,
	]);

	const value = useMemo(
		(): PaperEditorContextValue => ({
			blocks,
			dispatch,
			focusedBlockId,
			setFocusedBlockId,
			metadataForm,
			updateText,
			updateLatex,
			insertParagraphAfter,
			insertEquationAfter,
			insertHeadingAfter,
			removeBlockAndFocusPrevious,
			setBlockKind,
			focusBlock,
			registerBlockAnchor,
			scrollToBlock,
		}),
		[
			blocks,
			focusedBlockId,
			metadataForm,
			updateText,
			updateLatex,
			insertParagraphAfter,
			insertEquationAfter,
			insertHeadingAfter,
			removeBlockAndFocusPrevious,
			setBlockKind,
			focusBlock,
			registerBlockAnchor,
			scrollToBlock,
		],
	);

	return (
		<PaperEditorContext.Provider value={value}>
			{children}
		</PaperEditorContext.Provider>
	);
}

export function usePaperEditor(): PaperEditorContextValue {
	const ctx = useContext(PaperEditorContext);
	if (!ctx) throw new Error("usePaperEditor must be used within PaperEditorProvider");
	return ctx;
}
