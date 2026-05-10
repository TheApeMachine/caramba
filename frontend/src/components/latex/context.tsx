"use client";

import type React from "react";
import {
	createContext,
	useCallback,
	useContext,
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
import type { HeadingLevel, PaperBlock } from "#/components/latex/model/types";

type PaperEditorContextValue = {
	blocks: PaperBlock[];
	dispatch: React.Dispatch<PaperAction>;
	focusedBlockId: string | null;
	setFocusedBlockId: (id: string | null) => void;
	updateText: (id: string, text: string) => void;
	insertParagraphAfter: (afterId: string, text?: string) => void;
	insertHeadingAfter: (afterId: string, level: HeadingLevel) => void;
	removeBlockAndFocusPrevious: (id: string) => void;
	setBlockKind: (
		id: string,
		kind: "paragraph" | "heading",
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

	const insertParagraphAfter = useCallback(
		(afterId: string, text = "") => {
			const block: PaperBlock = {
				id: crypto.randomUUID(),
				type: "paragraph",
				text,
			};
			dispatch({ type: "INSERT_AFTER", afterId, block });
			queueMicrotask(() => focusBlock(block.id));
		},
		[focusBlock],
	);

	const insertHeadingAfter = useCallback(
		(afterId: string, level: HeadingLevel) => {
			const block: PaperBlock = {
				id: crypto.randomUUID(),
				type: "heading",
				level,
				text: "",
			};
			dispatch({ type: "INSERT_AFTER", afterId, block });
			queueMicrotask(() => focusBlock(block.id));
		},
		[focusBlock],
	);

	const removeBlockAndFocusPrevious = useCallback(
		(id: string) => {
			const idx = blocks.findIndex((b) => b.id === id);
			const prevId = idx > 0 ? blocks[idx - 1]?.id : undefined;
			dispatch({ type: "REMOVE_BLOCK", id });
			if (prevId) {
				queueMicrotask(() => focusBlock(prevId));
			}
		},
		[blocks, focusBlock],
	);

	const setBlockKind = useCallback(
		(id: string, kind: "paragraph" | "heading", level?: HeadingLevel) => {
			dispatch({ type: "SET_BLOCK_KIND", id, kind, level });
		},
		[],
	);

	const value = useMemo(
		(): PaperEditorContextValue => ({
			blocks,
			dispatch,
			focusedBlockId,
			setFocusedBlockId,
			updateText,
			insertParagraphAfter,
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
			updateText,
			insertParagraphAfter,
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
	if (!ctx) {
		throw new Error("usePaperEditor must be used within PaperEditorProvider");
	}
	return ctx;
}
