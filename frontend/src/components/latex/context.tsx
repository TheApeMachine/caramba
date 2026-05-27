"use client";

import { eq, useLiveQuery } from "@tanstack/react-db";
import { useStore } from "@tanstack/react-form";
import type React from "react";
import { createContext, useContext, useEffect, useMemo, useRef } from "react";
import {
	type ResearchPaperBlockRowType,
	researchPaperBlockCollection,
} from "#/collections/research_paper_blocks";
import {
	insertBlockAfter as insertBlockAfterAction,
	insertBlockAtStart as insertBlockAtStartAction,
	type InsertContext,
	insertEquationAfter as insertEquationAfterAction,
	insertHeadingAfter as insertHeadingAfterAction,
	insertListAfter as insertListAfterAction,
	insertParagraphAfter as insertParagraphAfterAction,
	removeBlock as removeBlockAction,
	reorderBlock as reorderBlockAction,
	setBlockKind as setBlockKindAction,
	updateBlockLatex,
	updateBlockText,
} from "#/components/latex/blocks/actions";
import type { SetBlockKindOptions } from "#/components/latex/blocks/convert-block";
import {
	setFocusedBlockId,
	useFocusedBlockId,
} from "#/components/latex/blocks/focus-store";
import { researchPaperBlockRowToBlock } from "#/components/latex/blocks/row-to-block";
import { editorBridge } from "#/components/latex/editor-bridge";
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
import { useResearchPaperSync } from "#/components/latex/paper-sync";

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
	insertBlockAtStart: (block: PaperBlock) => string;
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

const sortRows = (
	rows: ReadonlyArray<ResearchPaperBlockRowType>,
): ResearchPaperBlockRowType[] =>
	[...rows].sort((left, right) =>
		left.sort_order === right.sort_order
			? left.created_at.getTime() - right.created_at.getTime()
			: left.sort_order - right.sort_order,
	);

export const PaperEditorProvider = ({
	children,
	paperId: paperIdProp,
	bootstrapProjectId,
	onPaperBootstrapped,
}: {
	children: React.ReactNode;
	paperId?: string;
	bootstrapProjectId?: string;
	onPaperBootstrapped?: (paperId: string) => void;
}) => {
	const anchorsRef = useRef(new Map<string, HTMLElement>());
	const metadataForm = usePaperMetadataForm();

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
	} = useResearchPaperSync({
		paperIdProp,
		bootstrapProjectId,
		onPaperBootstrapped,
		metadata,
		metadataForm,
	});

	const blockQuery = useLiveQuery(
		(query) =>
			query
				.from({ row: researchPaperBlockCollection })
				.where(({ row }) => eq(row.paper_id, effectivePaperId ?? "")),
		[effectivePaperId],
	);

	const rows = useMemo(
		() => sortRows(blockQuery.data ?? []),
		[blockQuery.data],
	);

	const blocks = useMemo(
		() => rows.map(researchPaperBlockRowToBlock),
		[rows],
	);

	const focusedBlockId = useFocusedBlockId();

	const insertContext: InsertContext | null = useMemo(() => {
		if (!effectivePaperId) {
			return null;
		}

		const organizationSlug = rows[0]?.organization_slug ?? "";

		return {
			paperId: effectivePaperId,
			organizationSlug,
			blocks: rows,
		};
	}, [effectivePaperId, rows]);

	const focusBlock = (id: string): void => {
		const root = anchorsRef.current.get(id);
		const editable = root?.querySelector<HTMLElement>("[contenteditable]");
		editable?.focus();
		root?.scrollIntoView({ behavior: "smooth", block: "center" });
	};

	const registerBlockAnchor = (id: string, el: HTMLElement | null): void => {
		if (el) {
			anchorsRef.current.set(id, el);
			return;
		}

		anchorsRef.current.delete(id);
	};

	const insertParagraphAfter = (afterId: string, text = ""): string => {
		if (!insertContext) {
			return "";
		}

		const id = insertParagraphAfterAction(insertContext, afterId, text);
		queueMicrotask(() => focusBlock(id));
		return id;
	};

	const insertEquationAfter = (afterId: string, latex = ""): string => {
		if (!insertContext) {
			return "";
		}

		const id = insertEquationAfterAction(insertContext, afterId, latex);
		queueMicrotask(() => focusBlock(id));
		return id;
	};

	const insertHeadingAfter = (
		afterId: string,
		level: HeadingLevel,
	): string => {
		if (!insertContext) {
			return "";
		}

		const id = insertHeadingAfterAction(insertContext, afterId, level);
		queueMicrotask(() => focusBlock(id));
		return id;
	};

	const insertListAfter = (afterId: string, ordered: boolean): string => {
		if (!insertContext) {
			return "";
		}

		const id = insertListAfterAction(insertContext, afterId, ordered);
		queueMicrotask(() => focusBlock(id));
		return id;
	};

	const insertBlockAfter = (afterId: string, block: PaperBlock): string => {
		if (!insertContext) {
			return "";
		}

		const id = insertBlockAfterAction(insertContext, afterId, block);
		queueMicrotask(() => focusBlock(id));
		return id;
	};

	const insertBlockAtStart = (block: PaperBlock): string => {
		if (!insertContext) {
			return "";
		}

		const id = insertBlockAtStartAction(insertContext, block);
		queueMicrotask(() => focusBlock(id));
		return id;
	};

	const removeBlockAndFocusPrevious = (id: string): void => {
		if (!insertContext) {
			return;
		}

		const index = rows.findIndex((entry) => entry.id === id);
		const previousId = index > 0 ? rows[index - 1]?.id : undefined;

		removeBlockAction(insertContext, id);

		if (previousId) {
			queueMicrotask(() => focusBlock(previousId));
		}
	};

	const reorderBlock = (
		sourceId: string,
		targetId: string,
		position: "above" | "below",
	): void => {
		if (!insertContext) {
			return;
		}

		void reorderBlockAction(insertContext, sourceId, targetId, position);
	};

	const setBlockKind = (
		id: string,
		kind: PaperBlockKind,
		options?: SetBlockKindOptions,
	): void => {
		setBlockKindAction(id, kind, options);
	};

	const bridgeRef = useRef({
		blocks,
		insertParagraphAfter,
		insertHeadingAfter,
		insertEquationAfter,
		insertListAfter,
		insertBlockAfter,
		reorderBlock,
		setBlockKind,
		focusBlock,
		metadataForm,
	});

	bridgeRef.current = {
		blocks,
		insertParagraphAfter,
		insertHeadingAfter,
		insertEquationAfter,
		insertListAfter,
		insertBlockAfter,
		reorderBlock,
		setBlockKind,
		focusBlock,
		metadataForm,
	};

	const bridgePublishedRef = useRef(false);

	if (!bridgePublishedRef.current) {
		bridgePublishedRef.current = true;
		editorBridge.publish({
			getBlocks: () => bridgeRef.current.blocks,
			getMetadata: () =>
				bridgeRef.current.metadataForm.store.state.values as PaperMetadata,
			updateText: updateBlockText,
			updateLatex: updateBlockLatex,
			insertParagraphAfter: (afterId, text) =>
				bridgeRef.current.insertParagraphAfter(afterId, text),
			insertHeadingAfter: (afterId, level) =>
				bridgeRef.current.insertHeadingAfter(afterId, level),
			insertEquationAfter: (afterId, latex) =>
				bridgeRef.current.insertEquationAfter(afterId, latex),
			insertListAfter: (afterId, ordered) =>
				bridgeRef.current.insertListAfter(afterId, ordered),
			insertBlockAfter: (afterId, block) =>
				bridgeRef.current.insertBlockAfter(afterId, block),
			removeBlock: (id) => researchPaperBlockCollection.delete(id),
			reorderBlock: (sourceId, targetId, position) =>
				bridgeRef.current.reorderBlock(sourceId, targetId, position),
			setBlockKind: (id, kind, options) =>
				bridgeRef.current.setBlockKind(id, kind, options),
			updateMetadata: (patch) => {
				for (const [key, value] of Object.entries(patch)) {
					bridgeRef.current.metadataForm.setFieldValue(
						key as keyof PaperMetadata,
						value as string,
					);
				}
			},
			scrollToBlock: (id) => bridgeRef.current.focusBlock(id),
		});
	}

	useEffect(
		() => () => {
			editorBridge.publish(null);
			bridgePublishedRef.current = false;
		},
		[],
	);

	const paperPersistence: PaperEditorPersistence = {
		enabled: persistEnabled,
		ready: persistReady,
		waitingForRemote,
		bootstrapError,
		saveError,
		effectivePaperId,
	};

	const value: PaperEditorContextValue = {
		blocks,
		focusedBlockId,
		setFocusedBlockId,
		metadataForm,
		paperPersistence,
		updateText: updateBlockText,
		updateLatex: updateBlockLatex,
		insertParagraphAfter,
		insertEquationAfter,
		insertHeadingAfter,
		insertListAfter,
		insertBlockAfter,
		insertBlockAtStart,
		removeBlockAndFocusPrevious,
		reorderBlock,
		setBlockKind,
		focusBlock,
		registerBlockAnchor,
		scrollToBlock: focusBlock,
	};

	return (
		<PaperEditorContext.Provider value={value}>
			{children}
		</PaperEditorContext.Provider>
	);
};

export const usePaperEditor = (): PaperEditorContextValue => {
	const ctx = useContext(PaperEditorContext);

	if (!ctx) {
		throw new Error("usePaperEditor must be used within PaperEditorProvider");
	}

	return ctx;
};
