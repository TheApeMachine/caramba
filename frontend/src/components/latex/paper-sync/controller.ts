"use client";

import { Store } from "@tanstack/store";
import {
	coercePaperRevision,
	type ResearchPaperRowType,
	researchPaperCollection,
} from "#/collections/research_paper";
import {
	parsePaperDocument,
	serializePaperDocument,
} from "#/components/latex/model/paper-document";
import { createInitialPaperBlocks } from "#/components/latex/model/paper-reducer";
import type { PaperMetadata } from "#/components/latex/model/types";
import { PaperAutosave } from "#/components/latex/paper-sync/autosave";
import {
	clearBootstrapDraftId,
	readBootstrapDraftId,
	writeBootstrapDraftId,
} from "#/components/latex/paper-sync/bootstrap";
import {
	hasPaperIdProp,
	initialPaperSyncState,
	type PaperCollectionPort,
	type PaperSyncConfig,
	type PaperSyncState,
} from "#/components/latex/paper-sync/controller-types";
import { paperDocumentSnapshot } from "#/components/latex/paper-sync/snapshot";

export type {
	PaperCollectionPort,
	PaperSyncConfig,
	PaperSyncDocument,
	PaperSyncState,
} from "#/components/latex/paper-sync/controller-types";

/*
PaperSyncController owns the bootstrap / hydrate / autosave state
machine for one paper editor. It subscribes directly to
researchPaperCollection (no useLiveQuery needed) and exposes its
reactive state through a Tanstack Store so the hook layer is pure
read-and-render. The autosave timer + flush + conflict retry live in
a composed PaperAutosave instance so this file stays focused on
lifecycle, bootstrap, and hydration.
*/
export class PaperSyncController {
	readonly store: Store<PaperSyncState>;

	private config: PaperSyncConfig;
	private readonly collection: PaperCollectionPort;
	private collectionUnsub: (() => void) | null = null;
	private readonly autosave: PaperAutosave;
	private bootstrapInFlight = false;

	constructor(config: PaperSyncConfig) {
		this.config = config;
		this.collection =
			config.collection ??
			(researchPaperCollection as unknown as PaperCollectionPort);
		this.store = new Store<PaperSyncState>(initialPaperSyncState);
		this.autosave = new PaperAutosave({
			collection: this.collection,
			store: this.store,
			getDocument: () => this.config.getDocument(),
			getPaperId: () => this.effectivePaperId,
		});
		this.subscribeToCollection();
		this.runBootstrap();
	}

	configure(config: PaperSyncConfig): void {
		const previous = this.config;
		this.config = config;

		if (previous.paperIdProp !== config.paperIdProp) {
			this.resetHydration();

			if (hasPaperIdProp(config.paperIdProp) && previous.bootstrapProjectId) {
				clearBootstrapDraftId(previous.bootstrapProjectId);
			}
		}

		if (
			previous.bootstrapProjectId !== config.bootstrapProjectId ||
			previous.paperIdProp !== config.paperIdProp
		) {
			this.runBootstrap();
		}
	}

	dispose(): void {
		this.autosave.dispose();
		this.collectionUnsub?.();
		this.collectionUnsub = null;
	}

	notifyDocument(): void {
		this.autosave.notify();
	}

	get effectivePaperId(): string | null {
		if (hasPaperIdProp(this.config.paperIdProp)) {
			return this.config.paperIdProp ?? null;
		}

		return this.store.state.bootstrappedId;
	}

	get persistEnabled(): boolean {
		return (
			hasPaperIdProp(this.config.paperIdProp) ||
			Boolean(this.config.bootstrapProjectId)
		);
	}

	get waitingForRemote(): boolean {
		const paperId = this.effectivePaperId;

		if (!paperId) {
			return false;
		}

		return this.collection.get(paperId) === undefined;
	}

	get ready(): boolean {
		if (!this.persistEnabled) {
			return true;
		}

		return (
			this.effectivePaperId !== null &&
			this.store.state.hydratedRevision !== null &&
			!this.waitingForRemote
		);
	}

	private resetHydration(): void {
		this.store.setState((previous) => ({
			...previous,
			hydratedRevision: null,
			saveError: null,
		}));
		this.autosave.reset();
	}

	private subscribeToCollection(): void {
		const subscription = this.collection.subscribeChanges(
			() => this.handleCollectionTick(),
			{ includeInitialState: true },
		);

		this.collectionUnsub = () => subscription.unsubscribe();
	}

	private handleCollectionTick(): void {
		this.runBootstrap();
		this.handleRemoteRow();
	}

	private collectProjectPapers(projectId: string): ResearchPaperRowType[] {
		const matches: ResearchPaperRowType[] = [];

		for (const row of this.collection.values()) {
			if (row.research_project_id === projectId) {
				matches.push(row);
			}
		}

		return matches;
	}

	private runBootstrap(): void {
		const { paperIdProp, bootstrapProjectId, onPaperBootstrapped } =
			this.config;

		if (hasPaperIdProp(paperIdProp) || !bootstrapProjectId) {
			return;
		}

		if (this.store.state.bootstrappedId || this.bootstrapInFlight) {
			return;
		}

		const existing = this.collectProjectPapers(bootstrapProjectId);

		if (existing.length === 1) {
			onPaperBootstrapped?.(existing[0].id);
			return;
		}

		if (existing.length > 1) {
			return;
		}

		const storedId = readBootstrapDraftId(bootstrapProjectId);

		if (storedId) {
			this.store.setState((previous) => ({
				...previous,
				bootstrappedId: storedId,
			}));
			onPaperBootstrapped?.(storedId);
			return;
		}

		this.bootstrapInFlight = true;
		const newId = crypto.randomUUID();
		writeBootstrapDraftId(bootstrapProjectId, newId);

		void this.insertBootstrapRow(bootstrapProjectId, newId);
	}

	private async insertBootstrapRow(
		projectId: string,
		newId: string,
	): Promise<void> {
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

			const transaction = this.collection.insert({
				id: newId,
				research_project_id: projectId,
				organization_slug: "",
				title: "Untitled paper",
				document,
				revision: 1,
				created_at: now,
				updated_at: now,
			});

			await transaction.isPersisted.promise;

			this.store.setState((previous) => ({
				...previous,
				bootstrappedId: newId,
				bootstrapError: null,
			}));
			this.config.onPaperBootstrapped?.(newId);
		} catch (cause) {
			clearBootstrapDraftId(projectId);
			const message = cause instanceof Error ? cause.message : String(cause);
			this.store.setState((previous) => ({
				...previous,
				bootstrapError: message,
			}));
		} finally {
			this.bootstrapInFlight = false;
		}
	}

	private handleRemoteRow(): void {
		const paperId = this.effectivePaperId;

		if (!paperId) {
			return;
		}

		const row = this.collection.get(paperId);

		if (!row) {
			return;
		}

		const remoteRevision = coercePaperRevision(row.revision);
		this.autosave.bumpRevision(remoteRevision);
		const { hydratedRevision } = this.store.state;

		if (hydratedRevision === null) {
			this.hydrate(row);
			return;
		}

		if (remoteRevision <= hydratedRevision) {
			return;
		}

		if (this.autosave.isLocallyDirty()) {
			this.store.setState((previous) => ({
				...previous,
				hydratedRevision: remoteRevision,
			}));
			void this.autosave.flush();
			return;
		}

		this.hydrate(row);
	}

	private hydrate(row: ResearchPaperRowType): void {
		const parsed = parsePaperDocument(row.document);

		if (!parsed) {
			return;
		}

		const snapshot = paperDocumentSnapshot(parsed.metadata, parsed.blocks);
		this.autosave.markPersisted(row, snapshot);
		this.store.setState((previous) => ({
			...previous,
			hydratedRevision: coercePaperRevision(row.revision),
		}));

		this.config.applyDocument({
			blocks: parsed.blocks,
			metadata: parsed.metadata,
		});
	}
}

export const createPaperSyncController = (
	config: PaperSyncConfig,
): PaperSyncController => new PaperSyncController(config);
