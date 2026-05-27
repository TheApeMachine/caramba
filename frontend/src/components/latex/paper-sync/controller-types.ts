import type { ResearchPaperRowType } from "#/collections/research_paper";
import type { PaperMetadata } from "#/components/latex/model/types";

/*
PaperCollectionPort isolates the subset of Tanstack DB collection
behaviour the controller actually depends on. Production callers pass
researchPaperCollection; tests pass an in-memory fake.
*/
export type PaperCollectionPort = {
	get: (id: string) => ResearchPaperRowType | undefined;
	values: () => Iterable<ResearchPaperRowType>;
	subscribeChanges: (
		callback: () => void,
		options?: { includeInitialState?: boolean },
	) => { unsubscribe: () => void };
	insert: (row: ResearchPaperRowType) => {
		isPersisted: { promise: Promise<unknown> };
	};
	update: (
		id: string,
		options: { metadata?: unknown },
		draftFn: (draft: ResearchPaperRowType) => void,
	) => { isPersisted: { promise: Promise<unknown> } };
};

export type PaperSyncConfig = {
	paperIdProp?: string;
	bootstrapProjectId?: string;
	onPaperBootstrapped?: (paperId: string) => void;
	/** Snapshot accessor — the controller asks the editor for its
	 *  current metadata whenever it needs to compare or persist. The
	 *  controller never caches metadata across calls. */
	getMetadata: () => PaperMetadata;
	/** Sink applied when the controller hydrates from the remote row.
	 *  The caller is responsible for pushing the metadata fields into
	 *  its form. */
	applyMetadata: (metadata: PaperMetadata) => void;
	/** Optional hook fired the first time the controller materializes
	 *  a brand-new paper. The caller seeds the initial blocks. */
	onBootstrapPaperCreated?: (paperId: string) => void;
	collection?: PaperCollectionPort;
};

export type PaperSyncState = {
	bootstrappedId: string | null;
	bootstrapError: string | null;
	saveError: string | null;
	hydratedRevision: number | null;
};

export const initialPaperSyncState: PaperSyncState = {
	bootstrappedId: null,
	bootstrapError: null,
	saveError: null,
	hydratedRevision: null,
};

export const hasPaperIdProp = (value: string | undefined): boolean =>
	typeof value === "string" && value.trim() !== "";
