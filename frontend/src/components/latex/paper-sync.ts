"use client";

import { useLiveQuery } from "@tanstack/react-db";
import {
	type Dispatch,
	type MutableRefObject,
	useCallback,
	useEffect,
	useMemo,
	useRef,
	useState,
} from "react";
import {
	type ResearchPaperRowType,
	researchPaperCollection,
} from "#/collections/research_paper";
import {
	parsePaperDocument,
	serializePaperDocument,
} from "#/components/latex/model/paper-document";
import type { PaperAction } from "#/components/latex/model/paper-reducer";
import { createInitialPaperBlocks } from "#/components/latex/model/paper-reducer";
import type { PaperBlock, PaperMetadata } from "#/components/latex/model/types";

const AUTOSAVE_MS = 1200;

const DRAFT_STORAGE_PREFIX = "caramba:research-paper-bootstrap:";

const UUID_RE =
	/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function readBootstrapDraftId(projectId: string): string | null {
	if (typeof window === "undefined") {
		return null;
	}

	const raw = window.sessionStorage.getItem(
		`${DRAFT_STORAGE_PREFIX}${projectId}`,
	);

	if (!raw || !UUID_RE.test(raw)) {
		return null;
	}

	return raw;
}

type MetadataWriter = {
	setFieldValue: (name: keyof PaperMetadata, value: string) => void;
};

/*
useResearchPaperCollectionSync wires the paper reducer + metadata form to
researchPaperCollection: bootstrap row, hydrate from Electric, debounced autosave.
*/
export function useResearchPaperCollectionSync({
	paperIdProp,
	bootstrapProjectId,
	onPaperBootstrapped,
	dispatch,
	blocksRef,
	blocks,
	metadata,
	metadataForm,
}: {
	paperIdProp?: string;
	bootstrapProjectId?: string;
	onPaperBootstrapped?: (paperId: string) => void;
	dispatch: Dispatch<PaperAction>;
	blocksRef: MutableRefObject<PaperBlock[]>;
	blocks: PaperBlock[];
	metadata: PaperMetadata;
	metadataForm: MetadataWriter;
}): {
	effectivePaperId: string | null;
	persistEnabled: boolean;
	ready: boolean;
	waitingForRemote: boolean;
	bootstrapError: string | null;
	saveError: string | null;
} {
	const [bootstrappedId, setBootstrappedId] = useState<string | null>(null);
	const [bootstrapError, setBootstrapError] = useState<string | null>(null);
	const [saveError, setSaveError] = useState<string | null>(null);
	const [hydratedRevision, setHydratedRevision] = useState<number | null>(null);

	const effectivePaperId =
		paperIdProp && paperIdProp.trim() !== ""
			? paperIdProp
			: (bootstrappedId ?? null);

	const persistEnabled = Boolean(
		(paperIdProp && paperIdProp.trim() !== "") || bootstrapProjectId,
	);

	const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	const autosaveGenerationRef = useRef(0);

	// biome-ignore lint/correctness/useExhaustiveDependencies: reset hydration when resolved paper id changes
	useEffect(() => {
		setHydratedRevision(null);
		setSaveError(null);
	}, [effectivePaperId]);

	useEffect(() => {
		if (!bootstrapProjectId) {
			return;
		}

		if (!paperIdProp || paperIdProp.trim() === "") {
			return;
		}

		if (typeof window === "undefined") {
			return;
		}

		window.sessionStorage.removeItem(
			`${DRAFT_STORAGE_PREFIX}${bootstrapProjectId}`,
		);
	}, [bootstrapProjectId, paperIdProp]);

	const papersQuery = useLiveQuery(
		(query) => query.from({ row: researchPaperCollection }),
		[],
	);

	const remoteRow = useMemo(() => {
		const list = papersQuery.data as ResearchPaperRowType[] | undefined;

		if (!effectivePaperId || !list) {
			return undefined;
		}

		return list.find((row) => row.id === effectivePaperId);
	}, [papersQuery.data, effectivePaperId]);

	useEffect(() => {
		const hasPaperIdInUrl = Boolean(paperIdProp?.trim());

		if (hasPaperIdInUrl || bootstrappedId || !bootstrapProjectId) {
			return;
		}

		const storedId = readBootstrapDraftId(bootstrapProjectId);

		if (storedId) {
			setBootstrappedId(storedId);
			onPaperBootstrapped?.(storedId);

			return;
		}

		let cancelled = false;
		setBootstrapError(null);
		const newId = crypto.randomUUID();

		if (typeof window !== "undefined") {
			window.sessionStorage.setItem(
				`${DRAFT_STORAGE_PREFIX}${bootstrapProjectId}`,
				newId,
			);
		}

		void (async () => {
			try {
				const initialBlocks = createInitialPaperBlocks();
				const initialMetadata: PaperMetadata = {
					title: "",
					authors: "",
					keywords: "",
					abstract: "",
				};
				const document = serializePaperDocument(initialMetadata, initialBlocks);
				const now = new Date();

				const tx = researchPaperCollection.insert({
					id: newId,
					research_project_id: bootstrapProjectId,
					organization_slug: "",
					title: "Untitled paper",
					document,
					revision: 1,
					created_at: now,
					updated_at: now,
				});

				await tx.isPersisted.promise;

				if (cancelled) {
					return;
				}

				setBootstrappedId(newId);
				onPaperBootstrapped?.(newId);
			} catch (err) {
				if (typeof window !== "undefined") {
					window.sessionStorage.removeItem(
						`${DRAFT_STORAGE_PREFIX}${bootstrapProjectId}`,
					);
				}

				if (cancelled) {
					return;
				}

				const message = err instanceof Error ? err.message : String(err);
				setBootstrapError(message);
			}
		})();

		return () => {
			cancelled = true;
		};
	}, [paperIdProp, bootstrappedId, bootstrapProjectId, onPaperBootstrapped]);

	const hydrateFromRemote = useCallback(
		(row: ResearchPaperRowType) => {
			const parsed = parsePaperDocument(row.document);

			if (!parsed) {
				return;
			}

			setHydratedRevision(row.revision);
			dispatch({ type: "REPLACE_BLOCKS", blocks: parsed.blocks });
			metadataForm.setFieldValue("title", parsed.metadata.title);
			metadataForm.setFieldValue("authors", parsed.metadata.authors);
			metadataForm.setFieldValue("keywords", parsed.metadata.keywords);
			metadataForm.setFieldValue("abstract", parsed.metadata.abstract);
		},
		[dispatch, metadataForm],
	);

	useEffect(() => {
		if (!remoteRow || remoteRow.id !== effectivePaperId) {
			return;
		}

		if (hydratedRevision === null) {
			hydrateFromRemote(remoteRow);

			return;
		}

		if (remoteRow.revision > hydratedRevision) {
			hydrateFromRemote(remoteRow);
		}
	}, [effectivePaperId, remoteRow, hydratedRevision, hydrateFromRemote]);

	const flushSave = useCallback(async () => {
		if (!effectivePaperId || !remoteRow || remoteRow.id !== effectivePaperId) {
			return;
		}

		if (hydratedRevision === null) {
			return;
		}

		const blocksSnapshot = blocksRef.current;
		const document = serializePaperDocument(metadata, blocksSnapshot);
		const titleFromMeta = metadata.title.trim();
		const firstHeading = blocksSnapshot.find(
			(block) => block.type === "heading",
		);
		const title =
			titleFromMeta ||
			(firstHeading?.type === "heading" ? firstHeading.text.trim() : "") ||
			"Untitled paper";

		try {
			setSaveError(null);

			await researchPaperCollection.update(
				effectivePaperId,
				{ summary: "autosave" },
				(draft: ResearchPaperRowType) => {
					draft.title = title;
					draft.document = document;
					draft.updated_at = new Date();
				},
			);
		} catch (err) {
			const message = err instanceof Error ? err.message : String(err);
			setSaveError(message);

			if (message.includes("revision conflict") || message.includes("(409)")) {
				setHydratedRevision(null);
			}
		}
	}, [effectivePaperId, remoteRow, hydratedRevision, metadata, blocksRef]);

	// biome-ignore lint/correctness/useExhaustiveDependencies: document edits must restart autosave debounce
	useEffect(() => {
		if (!effectivePaperId || hydratedRevision === null) {
			return;
		}

		if (saveTimerRef.current !== null) {
			clearTimeout(saveTimerRef.current);
		}

		autosaveGenerationRef.current += 1;
		const generation = autosaveGenerationRef.current;

		saveTimerRef.current = setTimeout(() => {
			saveTimerRef.current = null;

			if (autosaveGenerationRef.current !== generation) {
				return;
			}

			void flushSave();
		}, AUTOSAVE_MS);

		return () => {
			if (saveTimerRef.current !== null) {
				clearTimeout(saveTimerRef.current);
				saveTimerRef.current = null;
			}
		};
	}, [blocks, metadata, effectivePaperId, hydratedRevision, flushSave]);

	const waitingForRemote = Boolean(effectivePaperId) && remoteRow === undefined;

	const ready =
		!persistEnabled ||
		(Boolean(effectivePaperId) &&
			hydratedRevision !== null &&
			remoteRow !== undefined);

	return {
		effectivePaperId,
		persistEnabled,
		ready,
		waitingForRemote,
		bootstrapError,
		saveError,
	};
}
