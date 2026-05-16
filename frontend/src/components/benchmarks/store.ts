import type { BenchmarkSpec } from "./model";

/*
RunRecord captures everything needed to render a saved or in-flight run in
the history view. Persisted to localStorage so the user can leave and come
back without losing context. Real runs would live in a server-side
collection; this is the fixture surface for now.
*/
export interface RunRecord {
	id: string;
	createdAt: number;
	spec: BenchmarkSpec;
	finalAccuracy: number | null;
	finalLoss: number | null;
	status: "queued" | "running" | "done" | "failed";
	durationSeconds: number | null;
}

const KEY = "caramba.benchmarks.runs";

const safeWindow = (): Storage | null => {
	if (typeof window === "undefined") return null;
	try {
		return window.localStorage;
	} catch {
		return null;
	}
};

export const loadRuns = (): RunRecord[] => {
	const storage = safeWindow();
	if (!storage) return [];
	const raw = storage.getItem(KEY);
	if (!raw) return [];
	try {
		const parsed = JSON.parse(raw);
		if (!Array.isArray(parsed)) return [];
		return parsed as RunRecord[];
	} catch {
		return [];
	}
};

export const saveRun = (record: RunRecord): void => {
	const storage = safeWindow();
	if (!storage) return;
	const existing = loadRuns().filter((entry) => entry.id !== record.id);
	const next = [record, ...existing].slice(0, 50);
	storage.setItem(KEY, JSON.stringify(next));
};

export const deleteRun = (id: string): void => {
	const storage = safeWindow();
	if (!storage) return;
	const next = loadRuns().filter((entry) => entry.id !== id);
	storage.setItem(KEY, JSON.stringify(next));
};
