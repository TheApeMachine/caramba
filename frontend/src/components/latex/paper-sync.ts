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
	coercePaperRevision,
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
import { ResearchPaperRevisionConflictError } from "#/server/research-papers";

const AUTOSAVE_MS = 1200;
const STRUCTURAL_AUTOSAVE_MS = 200;

const DRAFT_STORAGE_PREFIX = "caramba:research-paper-bootstrap:";

const paperDocumentSnapshot = (
	documentMetadata: PaperMetadata,
	documentBlocks: PaperBlock[],
): string =>
	JSON.stringify(serializePaperDocument(documentMetadata, documentBlocks));

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
	const saveInFlightRef = useRef(false);
	const persistedRevisionRef = useRef<number | null>(null);
	const lastPersistedSnapshotRef = useRef<string | null>(null);
	const blockStructureRef = useRef("");

	const rememberPersistedSnapshot = useCallback(
		(documentMetadata: PaperMetadata, documentBlocks: PaperBlock[]) => {
			lastPersistedSnapshotRef.current = paperDocumentSnapshot(
				documentMetadata,
				documentBlocks,
			);
		},
		[],
	);

	const isLocallyDirty = useCallback(() => {
		if (lastPersistedSnapshotRef.current === null) {
			return false;
		}

		return (
			paperDocumentSnapshot(metadata, blocksRef.current) !==
			lastPersistedSnapshotRef.current
		);
	}, [metadata, blocksRef]);

	// biome-ignore lint/correctness/useExhaustiveDependencies: reset hydration when resolved paper id changes
	useEffect(() => {
		setHydratedRevision(null);
		setSaveError(null);
		persistedRevisionRef.current = null;
		lastPersistedSnapshotRef.current = null;
		blockStructureRef.current = "";
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

	const projectPapers = useMemo(() => {
		const list = papersQuery.data as ResearchPaperRowType[] | undefined;

		if (!bootstrapProjectId || !list) {
			return [];
		}

		return list.filter((row) => row.research_project_id === bootstrapProjectId);
	}, [papersQuery.data, bootstrapProjectId]);

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

		if (projectPapers.length > 0) {
			if (projectPapers.length === 1) {
				onPaperBootstrapped?.(projectPapers[0].id);
			}

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
	}, [
		paperIdProp,
		bootstrappedId,
		bootstrapProjectId,
		onPaperBootstrapped,
		projectPapers,
	]);

	const hydrateFromRemote = useCallback(
		(row: ResearchPaperRowType) => {
			const parsed = parsePaperDocument(row.document);

			if (!parsed) {
				return;
			}

			persistedRevisionRef.current = coercePaperRevision(row.revision);
			setHydratedRevision(coercePaperRevision(row.revision));
			rememberPersistedSnapshot(parsed.metadata, parsed.blocks);
			dispatch({ type: "REPLACE_BLOCKS", blocks: parsed.blocks });
			metadataForm.setFieldValue("title", parsed.metadata.title);
			metadataForm.setFieldValue("authors", parsed.metadata.authors);
			metadataForm.setFieldValue("keywords", parsed.metadata.keywords);
			metadataForm.setFieldValue("abstract", parsed.metadata.abstract);
		},
		[dispatch, metadataForm, rememberPersistedSnapshot],
	);

	useEffect(() => {
		if (remoteRow?.revision === undefined) {
			return;
		}

		const remoteRevision = coercePaperRevision(remoteRow.revision);

		persistedRevisionRef.current = Math.max(
			persistedRevisionRef.current ?? 0,
			remoteRevision,
		);
	}, [remoteRow?.revision]);

	const flushSave = useCallback(
		async (retryRevision?: number) => {
			if (
				!effectivePaperId ||
				!remoteRow ||
				remoteRow.id !== effectivePaperId
			) {
				return;
			}

			if (hydratedRevision === null) {
				return;
			}

			if (saveInFlightRef.current) {
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
			const expectedRevision =
				retryRevision ??
				persistedRevisionRef.current ??
				coercePaperRevision(remoteRow.revision);

			saveInFlightRef.current = true;

			try {
				setSaveError(null);

				const transaction = researchPaperCollection.update(
					effectivePaperId,
					{
						metadata: {
							summary: "autosave",
							expected_revision: expectedRevision,
						},
					},
					(draft) => {
						draft.title = title;
						draft.document = document;
						draft.updated_at = new Date();
						draft.revision = expectedRevision + 1;
					},
				);

				await transaction.isPersisted.promise;

				const nextRevision = expectedRevision + 1;
				persistedRevisionRef.current = nextRevision;
				setHydratedRevision(nextRevision);
				rememberPersistedSnapshot(metadata, blocksSnapshot);
			} catch (err) {
				const conflict = ResearchPaperRevisionConflictError.fromUnknown(err);

				if (conflict !== null && retryRevision === undefined) {
					persistedRevisionRef.current = conflict.serverRevision;
					setHydratedRevision(conflict.serverRevision);
					setSaveError(conflict.message);
					saveInFlightRef.current = false;
					await flushSave(conflict.serverRevision);

					return;
				}

				const message = err instanceof Error ? err.message : String(err);
				setSaveError(message);
			} finally {
				saveInFlightRef.current = false;
			}
		},
		[
			effectivePaperId,
			remoteRow,
			hydratedRevision,
			metadata,
			blocksRef,
			rememberPersistedSnapshot,
		],
	);

	useEffect(() => {
		if (!remoteRow || remoteRow.id !== effectivePaperId) {
			return;
		}

		if (hydratedRevision === null) {
			hydrateFromRemote(remoteRow);

			return;
		}

		const remoteRevision = coercePaperRevision(remoteRow.revision);

		if (remoteRevision > hydratedRevision) {
			if (isLocallyDirty()) {
				persistedRevisionRef.current = remoteRevision;
				setHydratedRevision(remoteRevision);
				void flushSave();

				return;
			}

			hydrateFromRemote(remoteRow);
		}
	}, [
		effectivePaperId,
		remoteRow,
		hydratedRevision,
		hydrateFromRemote,
		isLocallyDirty,
		flushSave,
	]);

	// biome-ignore lint/correctness/useExhaustiveDependencies: document edits must restart autosave debounce
	useEffect(() => {
		if (!effectivePaperId || hydratedRevision === null) {
			return;
		}

		const blockStructure = blocks.map((block) => block.id).join("|");
		const structureChanged =
			blockStructureRef.current !== "" &&
			blockStructureRef.current !== blockStructure;
		blockStructureRef.current = blockStructure;

		if (saveTimerRef.current !== null) {
			clearTimeout(saveTimerRef.current);
		}

		autosaveGenerationRef.current += 1;
		const generation = autosaveGenerationRef.current;
		const debounceMs = structureChanged ? STRUCTURAL_AUTOSAVE_MS : AUTOSAVE_MS;

		saveTimerRef.current = setTimeout(() => {
			saveTimerRef.current = null;

			if (autosaveGenerationRef.current !== generation) {
				return;
			}

			void flushSave();
		}, debounceMs);

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
