import type { ResearchPaperRowType } from "#/collections/research_paper";
import type { PaperBlock, PaperMetadata } from "#/components/latex/model/types";

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

export type PaperSyncDocument = {
	blocks: PaperBlock[];
	metadata: PaperMetadata;
};

export type PaperSyncConfig = {
	paperIdProp?: string;
	bootstrapProjectId?: string;
	onPaperBootstrapped?: (paperId: string) => void;
	getDocument: () => PaperSyncDocument;
	applyDocument: (document: PaperSyncDocument) => void;
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
