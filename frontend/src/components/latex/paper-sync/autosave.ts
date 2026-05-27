"use client";

import type { Store } from "@tanstack/store";
import {
	coercePaperRevision,
	type ResearchPaperRowType,
} from "#/collections/research_paper";
import { serializePaperDocument } from "#/components/latex/model/paper-document";
import type { PaperMetadata } from "#/components/latex/model/types";
import type {
	PaperCollectionPort,
	PaperSyncState,
} from "#/components/latex/paper-sync/controller-types";
import { paperMetadataSnapshot } from "#/components/latex/paper-sync/snapshot";
import { ResearchPaperRevisionConflictError } from "#/server/research-papers";

const AUTOSAVE_MS = 1200;

type AutosaveDeps = {
	collection: PaperCollectionPort;
	store: Store<PaperSyncState>;
	getMetadata: () => PaperMetadata;
	getPaperId: () => string | null;
};

const resolveTitle = (metadata: PaperMetadata): string => {
	const fromMeta = metadata.title.trim();

	if (fromMeta) {
		return fromMeta;
	}

	return "Untitled paper";
};

/*
PaperMetadataAutosave owns the debounced metadata save + conflict-retry
state machine for one paper. Blocks live in their own collection and are
not this class's concern; this only persists title / authors / keywords
/ abstract changes back to the paper row.
*/
export class PaperMetadataAutosave {
	private readonly deps: AutosaveDeps;

	private timer: ReturnType<typeof setTimeout> | null = null;
	private generation = 0;
	private inFlight = false;
	private persistedRevision: number | null = null;
	private lastPersistedSnapshot: string | null = null;
	private lastNotifiedSnapshot: string | null = null;

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
	}

	markPersisted(row: ResearchPaperRowType, snapshot: string): void {
		this.persistedRevision = coercePaperRevision(row.revision);
		this.lastPersistedSnapshot = snapshot;
		this.lastNotifiedSnapshot = snapshot;
	}

	bumpRevision(revision: number): void {
		this.persistedRevision = Math.max(this.persistedRevision ?? 0, revision);
	}

	isLocallyDirty(): boolean {
		if (this.lastPersistedSnapshot === null) {
			return false;
		}

		return (
			paperMetadataSnapshot(this.deps.getMetadata()) !==
			this.lastPersistedSnapshot
		);
	}

	notify(): void {
		const paperId = this.deps.getPaperId();
		const state = this.deps.store.state;

		if (!paperId || state.hydratedRevision === null) {
			return;
		}

		const snapshot = paperMetadataSnapshot(this.deps.getMetadata());

		if (snapshot === this.lastNotifiedSnapshot) {
			return;
		}

		this.lastNotifiedSnapshot = snapshot;
		this.schedule();
	}

	private cancelTimer(): void {
		if (this.timer !== null) {
			clearTimeout(this.timer);
			this.timer = null;
		}
	}

	private schedule(): void {
		this.cancelTimer();
		this.generation += 1;
		const generation = this.generation;

		this.timer = setTimeout(() => {
			this.timer = null;

			if (this.generation !== generation) {
				return;
			}

			void this.flush();
		}, AUTOSAVE_MS);
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

		const metadata = this.deps.getMetadata();
		const document = serializePaperDocument(metadata);
		const title = resolveTitle(metadata);
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
			const snapshot = paperMetadataSnapshot(metadata);
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
