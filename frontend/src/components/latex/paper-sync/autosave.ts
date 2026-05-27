"use client";

import type { Store } from "@tanstack/store";
import {
	coercePaperRevision,
	type ResearchPaperRowType,
} from "#/collections/research_paper";
import { serializePaperDocument } from "#/components/latex/model/paper-document";
import type { PaperBlock } from "#/components/latex/model/types";
import type {
	PaperCollectionPort,
	PaperSyncDocument,
	PaperSyncState,
} from "#/components/latex/paper-sync/controller-types";
import {
	paperDocumentSnapshot,
	paperStructureSignature,
} from "#/components/latex/paper-sync/snapshot";
import { ResearchPaperRevisionConflictError } from "#/server/research-papers";

const AUTOSAVE_MS = 1200;
const STRUCTURAL_AUTOSAVE_MS = 200;

type AutosaveDeps = {
	collection: PaperCollectionPort;
	store: Store<PaperSyncState>;
	getDocument: () => PaperSyncDocument;
	getPaperId: () => string | null;
};

const resolveTitle = (
	metadata: PaperSyncDocument["metadata"],
	blocks: PaperBlock[],
): string => {
	const fromMeta = metadata.title.trim();

	if (fromMeta) {
		return fromMeta;
	}

	const heading = blocks.find((block) => block.type === "heading");

	if (heading?.type === "heading") {
		const headingText = heading.text.trim();

		if (headingText) {
			return headingText;
		}
	}

	return "Untitled paper";
};

/*
PaperAutosave owns the debounced autosave + conflict-retry state
machine for a single paper editor. It exposes notify() (snapshot
comparison + debounce), flush() (immediate save), and tracks the
persisted revision in private fields so the controller stays focused
on lifecycle and bootstrap concerns.
*/
export class PaperAutosave {
	private readonly deps: AutosaveDeps;

	private timer: ReturnType<typeof setTimeout> | null = null;
	private generation = 0;
	private inFlight = false;
	private persistedRevision: number | null = null;
	private lastPersistedSnapshot: string | null = null;
	private lastNotifiedSnapshot: string | null = null;
	private blockStructure = "";

	constructor(deps: AutosaveDeps) {
		this.deps = deps;
	}

	dispose(): void {
		this.cancelTimer();
	}

	reset(): void {
		this.cancelTimer();
		this.persistedRevision = null;
		this.lastPersistedSnapshot = null;
		this.lastNotifiedSnapshot = null;
		this.blockStructure = "";
	}

	markPersisted(row: ResearchPaperRowType, snapshot: string): void {
		this.persistedRevision = coercePaperRevision(row.revision);
		this.lastPersistedSnapshot = snapshot;
		this.lastNotifiedSnapshot = snapshot;
		this.blockStructure = "";
	}

	bumpRevision(revision: number): void {
		this.persistedRevision = Math.max(this.persistedRevision ?? 0, revision);
	}

	getPersistedRevision(): number | null {
		return this.persistedRevision;
	}

	isLocallyDirty(): boolean {
		if (this.lastPersistedSnapshot === null) {
			return false;
		}

		const document = this.deps.getDocument();

		return (
			paperDocumentSnapshot(document.metadata, document.blocks) !==
			this.lastPersistedSnapshot
		);
	}

	notify(): void {
		const paperId = this.deps.getPaperId();
		const state = this.deps.store.state;

		if (!paperId || state.hydratedRevision === null) {
			return;
		}

		const document = this.deps.getDocument();
		const snapshot = paperDocumentSnapshot(document.metadata, document.blocks);

		if (snapshot === this.lastNotifiedSnapshot) {
			return;
		}

		this.lastNotifiedSnapshot = snapshot;
		this.schedule(document.blocks);
	}

	private cancelTimer(): void {
		if (this.timer !== null) {
			clearTimeout(this.timer);
			this.timer = null;
		}
	}

	private schedule(blocks: PaperBlock[]): void {
		const signature = paperStructureSignature(blocks);
		const structureChanged =
			this.blockStructure !== "" && this.blockStructure !== signature;
		this.blockStructure = signature;

		this.cancelTimer();
		this.generation += 1;
		const generation = this.generation;
		const delay = structureChanged ? STRUCTURAL_AUTOSAVE_MS : AUTOSAVE_MS;

		this.timer = setTimeout(() => {
			this.timer = null;

			if (this.generation !== generation) {
				return;
			}

			void this.flush();
		}, delay);
	}

	async flush(retryRevision?: number): Promise<void> {
		const paperId = this.deps.getPaperId();

		if (!paperId) {
			return;
		}

		const row = this.deps.collection.get(paperId);

		if (!row || row.id !== paperId) {
			return;
		}

		if (this.deps.store.state.hydratedRevision === null || this.inFlight) {
			return;
		}

		const { blocks, metadata } = this.deps.getDocument();
		const document = serializePaperDocument(metadata, blocks);
		const title = resolveTitle(metadata, blocks);
		const expectedRevision =
			retryRevision ??
			this.persistedRevision ??
			coercePaperRevision(row.revision);

		this.inFlight = true;
		this.deps.store.setState((previous) => ({ ...previous, saveError: null }));

		try {
			const transaction = this.deps.collection.update(
				paperId,
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
			this.persistedRevision = nextRevision;
			const snapshot = paperDocumentSnapshot(metadata, blocks);
			this.lastPersistedSnapshot = snapshot;
			this.lastNotifiedSnapshot = snapshot;
			this.deps.store.setState((previous) => ({
				...previous,
				hydratedRevision: nextRevision,
			}));
		} catch (cause) {
			await this.handleSaveError(cause, retryRevision);
		} finally {
			this.inFlight = false;
		}
	}

	private async handleSaveError(
		cause: unknown,
		retryRevision: number | undefined,
	): Promise<void> {
		const conflict = ResearchPaperRevisionConflictError.fromUnknown(cause);

		if (conflict !== null && retryRevision === undefined) {
			this.persistedRevision = conflict.serverRevision;
			this.deps.store.setState((previous) => ({
				...previous,
				hydratedRevision: conflict.serverRevision,
				saveError: conflict.message,
			}));
			this.inFlight = false;
			await this.flush(conflict.serverRevision);
			return;
		}

		const message = cause instanceof Error ? cause.message : String(cause);
		this.deps.store.setState((previous) => ({
			...previous,
			saveError: message,
		}));
	}
}
