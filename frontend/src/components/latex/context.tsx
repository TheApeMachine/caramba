"use client";

import { useStore } from "@tanstack/react-form";
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
	type SetBlockKindOptions,
} from "#/components/latex/model/paper-reducer";
import type {
	HeadingLevel,
	PaperBlock,
	PaperBlockKind,
	PaperMetadata,
} from "#/components/latex/model/types";
import {
	type PaperMetadataFormApi,
	usePaperMetadataForm,
} from "#/components/latex/panels/metadata-tab";
import { useResearchPaperCollectionSync } from "#/components/latex/paper-sync";
import { editorBridge } from "./editor-bridge";

export type PaperEditorPersistence = {
	enabled: boolean;
	ready: boolean;
	waitingForRemote: boolean;
	bootstrapError: string | null;
	saveError: string | null;
	effectivePaperId: string | null;
};

type PaperEditorContextValue = {
	blocks: PaperBlock[];
	dispatch: React.Dispatch<PaperAction>;
	focusedBlockId: string | null;
	setFocusedBlockId: (id: string | null) => void;
	metadataForm: PaperMetadataFormApi;
	paperPersistence: PaperEditorPersistence;
	updateText: (id: string, text: string) => void;
	updateLatex: (id: string, latex: string) => void;
	insertParagraphAfter: (afterId: string, text?: string) => string;
	insertEquationAfter: (afterId: string, latex?: string) => string;
	insertHeadingAfter: (afterId: string, level: HeadingLevel) => string;
	insertListAfter: (afterId: string, ordered: boolean) => string;
	insertBlockAfter: (afterId: string, block: PaperBlock) => string;
	removeBlockAndFocusPrevious: (id: string) => void;
	reorderBlock: (
		sourceId: string,
		targetId: string,
		position: "above" | "below",
	) => void;
	setBlockKind: (
		id: string,
		kind: PaperBlockKind,
		options?: SetBlockKindOptions,
	) => void;
	focusBlock: (id: string) => void;
	registerBlockAnchor: (id: string, el: HTMLElement | null) => void;
	scrollToBlock: (id: string) => void;
};

const PaperEditorContext = createContext<PaperEditorContextValue | null>(null);

function newId(): string {
	return crypto.randomUUID();
}

export function PaperEditorProvider({
	children,
	paperId: paperIdProp,
	bootstrapProjectId,
	onPaperBootstrapped,
}: {
	children: React.ReactNode;
	paperId?: string;
	bootstrapProjectId?: string;
	onPaperBootstrapped?: (paperId: string) => void;
}) {
	const [blocks, dispatch] = useReducer(
		paperReducer,
		undefined,
		createInitialPaperBlocks,
	);
	const [focusedBlockId, setFocusedBlockId] = useState<string | null>(null);
	const anchorsRef = useRef(new Map<string, HTMLElement>());
	const metadataForm = usePaperMetadataForm();

	const blocksRef = useRef(blocks);
	blocksRef.current = blocks;

	const metadata = useStore(
		metadataForm.store,
		(state) => state.values as PaperMetadata,
	);

	const {
		effectivePaperId,
		persistEnabled,
		ready: persistReady,
		waitingForRemote,
		bootstrapError,
		saveError,
	} = useResearchPaperCollectionSync({
		paperIdProp,
		bootstrapProjectId,
		onPaperBootstrapped,
		dispatch,
		blocksRef,
		blocks,
		metadata,
		metadataForm,
	});

	const paperPersistence = useMemo(
		(): PaperEditorPersistence => ({
			enabled: persistEnabled,
			ready: persistReady,
			waitingForRemote,
			bootstrapError,
			saveError,
			effectivePaperId,
		}),
		[
			persistEnabled,
			persistReady,
			waitingForRemote,
			bootstrapError,
			saveError,
			effectivePaperId,
		],
	);

	const registerBlockAnchor = useCallback(
		(id: string, el: HTMLElement | null) => {
			if (el) {
				anchorsRef.current.set(id, el);
				return;
			}

			anchorsRef.current.delete(id);
		},
		[],
	);

	const focusBlock = useCallback((id: string) => {
		const root = anchorsRef.current.get(id);
		const editable = root?.querySelector<HTMLElement>("[contenteditable]");
		editable?.focus();
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

	const insertBlockAfter = useCallback(
		(afterId: string, block: PaperBlock): string => {
			dispatch({ type: "INSERT_AFTER", afterId, block });
			queueMicrotask(() => focusBlock(block.id));
			return block.id;
		},
		[focusBlock],
	);

	const insertParagraphAfter = useCallback(
		(afterId: string, text = ""): string =>
			insertBlockAfter(afterId, { id: newId(), type: "paragraph", text }),
		[insertBlockAfter],
	);

	const insertEquationAfter = useCallback(
		(afterId: string, latex = ""): string =>
			insertBlockAfter(afterId, {
				id: newId(),
				type: "equation",
				latex,
				display: true,
			}),
		[insertBlockAfter],
	);

	const insertHeadingAfter = useCallback(
		(afterId: string, level: HeadingLevel): string =>
			insertBlockAfter(afterId, {
				id: newId(),
				type: "heading",
				level,
				text: "",
			}),
		[insertBlockAfter],
	);

	const insertListAfter = useCallback(
		(afterId: string, ordered: boolean): string =>
			insertBlockAfter(afterId, {
				id: newId(),
				type: "list",
				ordered,
				text: "",
			}),
		[insertBlockAfter],
	);

	const removeBlockAndFocusPrevious = useCallback(
		(id: string) => {
			const idx = blocksRef.current.findIndex((block) => block.id === id);
			const prevId = idx > 0 ? blocksRef.current[idx - 1]?.id : undefined;
			dispatch({ type: "REMOVE_BLOCK", id });

			if (prevId) {
				queueMicrotask(() => focusBlock(prevId));
			}
		},
		[focusBlock],
	);

	const reorderBlock = useCallback(
		(sourceId: string, targetId: string, position: "above" | "below") => {
			dispatch({ type: "REORDER_BLOCK", sourceId, targetId, position });
		},
		[],
	);

	const setBlockKind = useCallback(
		(id: string, kind: PaperBlockKind, options?: SetBlockKindOptions) => {
			dispatch({ type: "SET_BLOCK_KIND", id, kind, options });
		},
		[],
	);

	useEffect(() => {
		editorBridge.register({
			getBlocks: () => blocksRef.current,
			getMetadata: () => metadataForm.store.state.values as PaperMetadata,
			updateText,
			updateLatex,
			insertParagraphAfter,
			insertHeadingAfter,
			insertEquationAfter,
			insertListAfter,
			insertBlockAfter,
			removeBlock: (id) => dispatch({ type: "REMOVE_BLOCK", id }),
			reorderBlock,
			setBlockKind,
			updateMetadata: (patch) => {
				for (const [key, val] of Object.entries(patch)) {
					metadataForm.setFieldValue(key as keyof PaperMetadata, val as string);
				}
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
		insertListAfter,
		insertBlockAfter,
		reorderBlock,
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
			paperPersistence,
			updateText,
			updateLatex,
			insertParagraphAfter,
			insertEquationAfter,
			insertHeadingAfter,
			insertListAfter,
			insertBlockAfter,
			removeBlockAndFocusPrevious,
			reorderBlock,
			setBlockKind,
			focusBlock,
			registerBlockAnchor,
			scrollToBlock,
		}),
		[
			blocks,
			focusedBlockId,
			metadataForm,
			paperPersistence,
			updateText,
			updateLatex,
			insertParagraphAfter,
			insertEquationAfter,
			insertHeadingAfter,
			insertListAfter,
			insertBlockAfter,
			removeBlockAndFocusPrevious,
			reorderBlock,
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
