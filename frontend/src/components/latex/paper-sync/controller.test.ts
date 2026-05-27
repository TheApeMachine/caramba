import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { serializePaperDocument } from "#/components/latex/model/paper-document";
import { createPaperSyncController } from "#/components/latex/paper-sync/controller";
import {
	AUTOSAVE_DELAY_MS,
	buildBlocks,
	buildHarness,
	buildMetadata,
	buildRow,
	flush,
	stubCryptoUUID,
} from "#/components/latex/paper-sync/controller.test-helpers";

beforeEach(() => {
	if (typeof window !== "undefined") {
		window.sessionStorage.clear();
	}
	vi.useFakeTimers();
});

afterEach(() => {
	vi.useRealTimers();
});

describe("PaperSyncController", () => {
	it("returns immediately when neither paperIdProp nor bootstrapProjectId is set", () => {
		const harness = buildHarness();
		const controller = createPaperSyncController({
			getDocument: harness.getDocument,
			applyDocument: (doc) => harness.applied.push(doc),
			collection: harness.collection,
		});

		expect(controller.persistEnabled).toBe(false);
		expect(controller.ready).toBe(true);
		expect(controller.effectivePaperId).toBe(null);
		controller.dispose();
	});

	it("bootstraps a new paper when none exists for the project", async () => {
		const harness = buildHarness();
		const projectId = "22222222-2222-4222-8222-222222222222";
		const restore = stubCryptoUUID("33333333-3333-4333-8333-333333333333");

		const controller = createPaperSyncController({
			bootstrapProjectId: projectId,
			onPaperBootstrapped: (id) => harness.bootstrapped.push(id),
			getDocument: harness.getDocument,
			applyDocument: (doc) => harness.applied.push(doc),
			collection: harness.collection,
		});

		await vi.runAllTimersAsync();
		await flush();
		await flush();

		expect(harness.collection.insertCalls).toHaveLength(1);
		expect(harness.collection.insertCalls[0].research_project_id).toBe(projectId);
		expect(harness.bootstrapped).toEqual([
			"33333333-3333-4333-8333-333333333333",
		]);
		expect(controller.store.state.bootstrappedId).toBe(
			"33333333-3333-4333-8333-333333333333",
		);

		restore();
		controller.dispose();
	});

	it("reuses the single existing paper for a project without inserting", async () => {
		const harness = buildHarness();
		const projectId = "44444444-4444-4444-8444-444444444444";
		const existing = buildRow({
			id: "55555555-5555-4555-8555-555555555555",
			research_project_id: projectId,
		});
		harness.collection.rows.set(existing.id, existing);

		const controller = createPaperSyncController({
			bootstrapProjectId: projectId,
			onPaperBootstrapped: (id) => harness.bootstrapped.push(id),
			getDocument: harness.getDocument,
			applyDocument: (doc) => harness.applied.push(doc),
			collection: harness.collection,
		});

		await flush();

		expect(harness.collection.insertCalls).toHaveLength(0);
		expect(harness.bootstrapped).toEqual([existing.id]);
		controller.dispose();
	});

	it("hydrates the editor from the remote row and tracks revision", async () => {
		const harness = buildHarness();
		const paperId = "66666666-6666-4666-8666-666666666666";
		const remoteRow = buildRow({
			id: paperId,
			revision: 7,
			document: serializePaperDocument(
				{ ...buildMetadata(), title: "Remote title" },
				buildBlocks("hydrated"),
			),
		});
		harness.collection.rows.set(paperId, remoteRow);

		const controller = createPaperSyncController({
			paperIdProp: paperId,
			getDocument: harness.getDocument,
			applyDocument: (doc) => harness.applied.push(doc),
			collection: harness.collection,
		});

		await flush();

		expect(harness.applied).toHaveLength(1);
		expect(harness.applied[0].metadata.title).toBe("Remote title");
		expect(controller.store.state.hydratedRevision).toBe(7);
		expect(controller.ready).toBe(true);
		controller.dispose();
	});

	it("debounces autosave on document changes and writes with expected revision", async () => {
		const harness = buildHarness();
		const paperId = "77777777-7777-4777-8777-777777777777";
		harness.collection.rows.set(paperId, buildRow({ id: paperId, revision: 3 }));

		const controller = createPaperSyncController({
			paperIdProp: paperId,
			getDocument: harness.getDocument,
			applyDocument: (doc) => harness.applied.push(doc),
			collection: harness.collection,
		});

		await flush();

		harness.setDocument({
			blocks: buildBlocks("edited"),
			metadata: buildMetadata(),
		});
		controller.notifyDocument();

		await vi.advanceTimersByTimeAsync(AUTOSAVE_DELAY_MS - 1);
		expect(harness.collection.updateCalls).toHaveLength(0);

		await vi.advanceTimersByTimeAsync(2);
		await flush();

		expect(harness.collection.updateCalls).toHaveLength(1);
		expect(harness.collection.updateCalls[0].metadata).toEqual({
			summary: "autosave",
			expected_revision: 3,
		});
		expect(controller.store.state.hydratedRevision).toBe(4);
		controller.dispose();
	});

	it("retries the save with the server revision on conflict", async () => {
		const harness = buildHarness();
		const paperId = "88888888-8888-4888-8888-888888888888";
		harness.collection.rows.set(paperId, buildRow({ id: paperId, revision: 5 }));
		harness.collection.nextUpdateOutcome = "conflict";
		harness.collection.conflictServerRevision = 9;

		const controller = createPaperSyncController({
			paperIdProp: paperId,
			getDocument: harness.getDocument,
			applyDocument: (doc) => harness.applied.push(doc),
			collection: harness.collection,
		});

		await flush();

		harness.setDocument({
			blocks: buildBlocks("local edit"),
			metadata: buildMetadata(),
		});
		controller.notifyDocument();
		await vi.advanceTimersByTimeAsync(AUTOSAVE_DELAY_MS);
		await flush();
		await flush();

		expect(harness.collection.updateCalls).toHaveLength(2);
		expect(harness.collection.updateCalls[0].metadata).toEqual({
			summary: "autosave",
			expected_revision: 5,
		});
		expect(harness.collection.updateCalls[1].metadata).toEqual({
			summary: "autosave",
			expected_revision: 9,
		});
		expect(controller.store.state.hydratedRevision).toBe(10);
		controller.dispose();
	});

	it("captures save errors that are not revision conflicts", async () => {
		const harness = buildHarness();
		const paperId = "99999999-9999-4999-8999-999999999999";
		harness.collection.rows.set(paperId, buildRow({ id: paperId, revision: 2 }));
		harness.collection.nextUpdateOutcome = "error";

		const controller = createPaperSyncController({
			paperIdProp: paperId,
			getDocument: harness.getDocument,
			applyDocument: (doc) => harness.applied.push(doc),
			collection: harness.collection,
		});

		await flush();

		harness.setDocument({
			blocks: buildBlocks("edit"),
			metadata: buildMetadata(),
		});
		controller.notifyDocument();
		await vi.advanceTimersByTimeAsync(AUTOSAVE_DELAY_MS);
		await flush();

		expect(controller.store.state.saveError).toBe("save failed");
		controller.dispose();
	});

	it("ignores duplicate notifyDocument calls with the same snapshot", async () => {
		const harness = buildHarness();
		const paperId = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
		harness.collection.rows.set(paperId, buildRow({ id: paperId, revision: 1 }));

		const controller = createPaperSyncController({
			paperIdProp: paperId,
			getDocument: harness.getDocument,
			applyDocument: (doc) => harness.applied.push(doc),
			collection: harness.collection,
		});

		await flush();

		controller.notifyDocument();
		controller.notifyDocument();
		controller.notifyDocument();

		await vi.advanceTimersByTimeAsync(AUTOSAVE_DELAY_MS);
		await flush();

		expect(harness.collection.updateCalls).toHaveLength(0);
		controller.dispose();
	});
});
