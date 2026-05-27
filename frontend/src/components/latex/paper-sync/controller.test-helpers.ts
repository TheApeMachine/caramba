import type { ResearchPaperRowType } from "#/collections/research_paper";
import { serializePaperDocument } from "#/components/latex/model/paper-document";
import type { PaperMetadata } from "#/components/latex/model/types";
import type { PaperCollectionPort } from "#/components/latex/paper-sync/controller";
import { ResearchPaperRevisionConflictError } from "#/server/research-papers";

export const flush = (): Promise<void> =>
	new Promise((resolve) => setTimeout(resolve, 0));

export const buildMetadata = (
	overrides: Partial<PaperMetadata> = {},
): PaperMetadata => ({
	title: "",
	authors: "",
	keywords: "",
	abstract: "",
	...overrides,
});

export const buildRow = (
	overrides: Partial<ResearchPaperRowType>,
): ResearchPaperRowType => {
	const now = new Date();
	const metadata = buildMetadata();

	return {
		id: "00000000-0000-4000-8000-000000000001",
		research_project_id: "11111111-1111-4111-8111-111111111111",
		organization_slug: "",
		title: "Paper",
		document: serializePaperDocument(metadata),
		revision: 1,
		created_at: now,
		updated_at: now,
		...overrides,
	};
};

type Persistence = {
	persist: () => void;
	reject: (cause: unknown) => void;
	promise: Promise<unknown>;
};

const createPersistence = (): Persistence => {
	let persist: () => void = () => {};
	let reject: (cause: unknown) => void = () => {};

	const promise = new Promise((resolve, fail) => {
		persist = () => resolve(undefined);
		reject = (cause: unknown) => fail(cause);
	});

	return { persist, reject, promise };
};

export class FakeCollection implements PaperCollectionPort {
	rows = new Map<string, ResearchPaperRowType>();
	insertCalls: ResearchPaperRowType[] = [];
	updateCalls: Array<{
		id: string;
		metadata: unknown;
		next: ResearchPaperRowType;
		original: ResearchPaperRowType;
	}> = [];
	nextUpdateOutcome: "success" | "conflict" | "error" = "success";
	conflictServerRevision = 0;

	private listeners = new Set<() => void>();
	private pendingPersistence: Persistence[] = [];

	get = (id: string) => this.rows.get(id);

	values = () => this.rows.values();

	subscribeChanges = (callback: () => void) => {
		this.listeners.add(callback);
		callback();
		return { unsubscribe: () => this.listeners.delete(callback) };
	};

	insert = (row: ResearchPaperRowType) => {
		this.insertCalls.push(row);
		this.rows.set(row.id, row);
		const persistence = createPersistence();
		this.pendingPersistence.push(persistence);
		queueMicrotask(() => this.fireListeners());
		return { isPersisted: { promise: persistence.promise } };
	};

	update = (
		id: string,
		options: { metadata?: unknown },
		draftFn: (draft: ResearchPaperRowType) => void,
	) => {
		const existing = this.rows.get(id);

		if (!existing) {
			throw new Error(`No row for ${id}`);
		}

		const next: ResearchPaperRowType = JSON.parse(JSON.stringify(existing));
		next.created_at = new Date(next.created_at);
		next.updated_at = new Date(next.updated_at);
		draftFn(next);

		this.updateCalls.push({
			id,
			metadata: options.metadata,
			next,
			original: existing,
		});

		const persistence = createPersistence();

		if (this.nextUpdateOutcome === "conflict") {
			this.nextUpdateOutcome = "success";
			persistence.reject(
				new ResearchPaperRevisionConflictError(
					this.conflictServerRevision,
					`{"revision": ${this.conflictServerRevision}}`,
				),
			);
		} else if (this.nextUpdateOutcome === "error") {
			this.nextUpdateOutcome = "success";
			persistence.reject(new Error("save failed"));
		} else {
			this.rows.set(id, next);
			persistence.persist();
			queueMicrotask(() => this.fireListeners());
		}

		return { isPersisted: { promise: persistence.promise } };
	};

	persistAll(): void {
		for (const persistence of this.pendingPersistence) {
			persistence.persist();
		}
		this.pendingPersistence = [];
	}

	private fireListeners(): void {
		for (const listener of this.listeners) {
			listener();
		}
	}
}

export type ControllerHarness = {
	collection: FakeCollection;
	getMetadata: () => PaperMetadata;
	setMetadata: (metadata: PaperMetadata) => void;
	applied: PaperMetadata[];
	bootstrapped: string[];
	created: string[];
};

export const buildHarness = (): ControllerHarness => {
	let current: PaperMetadata = buildMetadata();
	const applied: PaperMetadata[] = [];
	const bootstrapped: string[] = [];
	const created: string[] = [];

	return {
		collection: new FakeCollection(),
		getMetadata: () => current,
		setMetadata: (next) => {
			current = next;
		},
		applied,
		bootstrapped,
		created,
	};
};

export const stubCryptoUUID = (value: string) => {
	const original = crypto.randomUUID;
	(crypto as { randomUUID: () => string }).randomUUID = () => value;

	return () => {
		(crypto as { randomUUID: () => string }).randomUUID = original;
	};
};

export const AUTOSAVE_DELAY_MS = 1200;
