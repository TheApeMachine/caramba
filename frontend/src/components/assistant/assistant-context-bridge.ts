export type SemanticContextEntry = {
	key: string;
	label: string;
	value: string;
	persistent?: boolean;
};

const entries = new Map<string, SemanticContextEntry>();
const subscribers = new Set<() => void>();
let cachedSnapshot: SemanticContextEntry[] = [];

function entriesEqual(
	left: SemanticContextEntry,
	right: SemanticContextEntry,
): boolean {
	return (
		left.key === right.key &&
		left.label === right.label &&
		left.value === right.value &&
		left.persistent === right.persistent
	);
}

function rebuildSnapshot(): void {
	cachedSnapshot = [...entries.values()];
}

function notify() {
	for (const subscriber of subscribers) {
		subscriber();
	}
}

/*
assistantContextBridge holds logical application context published by feature
providers. Values survive CSS visibility changes and replace DOM scraping as
the primary context source for the assistant.
*/
export const assistantContextBridge = {
	publish(entry: SemanticContextEntry) {
		const existing = entries.get(entry.key);

		if (existing && entriesEqual(existing, entry)) {
			return;
		}

		entries.set(entry.key, entry);
		rebuildSnapshot();
		notify();
	},
	unpublish(key: string) {
		if (!entries.has(key)) {
			return;
		}

		entries.delete(key);
		rebuildSnapshot();
		notify();
	},
	snapshot(): SemanticContextEntry[] {
		return cachedSnapshot;
	},
	subscribe(callback: () => void): () => void {
		subscribers.add(callback);
		return () => {
			subscribers.delete(callback);
		};
	},
};
