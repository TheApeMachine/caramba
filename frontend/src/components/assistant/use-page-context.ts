/*
usePageContext assembles assistant context from semantic state published by
feature providers, with DOM scraping as a secondary fallback for legacy
data-context markers.
*/
import { useCallback } from "react";
import { assistantContextBridge } from "./assistant-context-bridge";

const MAX_CHARS = 512;
const MAX_TOTAL_CHARS = 4096;

type ContextEntry = {
	label: string;
	key: string;
	value: string;
};

type ExtractionType = "text" | "value" | "json" | "count";

function isVisible(element: Element): boolean {
	const style = window.getComputedStyle(element);
	return (
		style.display !== "none" &&
		style.visibility !== "hidden" &&
		style.opacity !== "0"
	);
}

function extractValue(element: Element, type: ExtractionType): string {
	switch (type) {
		case "value": {
			const input = element as
				| HTMLInputElement
				| HTMLTextAreaElement
				| HTMLSelectElement;
			return input.value ?? "";
		}
		case "json": {
			const raw =
				element.getAttribute("data-context-value") ?? element.textContent ?? "";
			try {
				return JSON.stringify(JSON.parse(raw), null, 2);
			} catch {
				return raw;
			}
		}
		case "count": {
			const children = element.querySelectorAll("[data-context-item]");
			return children.length > 0
				? `${children.length} item${children.length === 1 ? "" : "s"}`
				: element.childElementCount.toString();
		}
		default: {
			return (element.textContent ?? "").replace(/\s+/g, " ").trim();
		}
	}
}

function scrapeDomContext(): ContextEntry[] {
	const nodes = document.querySelectorAll("[data-context]");
	const entries: ContextEntry[] = [];

	for (const node of nodes) {
		if (!isVisible(node)) {
			continue;
		}

		const label = node.getAttribute("data-context") ?? "";
		const key =
			node.getAttribute("data-context-key") ??
			label.toLowerCase().replace(/\s+/g, "_");
		const type = (node.getAttribute("data-context-type") ??
			"text") as ExtractionType;
		const raw = extractValue(node, type);
		const value = raw.length > MAX_CHARS ? `${raw.slice(0, MAX_CHARS)}…` : raw;

		if (value) {
			entries.push({ label, key, value });
		}
	}

	return entries;
}

function mergeEntries(entries: ContextEntry[]): ContextEntry[] {
	const merged = new Map<string, ContextEntry>();

	for (const entry of entries) {
		merged.set(entry.key, entry);
	}

	return [...merged.values()];
}

function format(entries: ContextEntry[], route: string): string {
	if (entries.length === 0) {
		return "";
	}

	const lines: string[] = [
		`[Page context — ${route}]`,
		...entries.map((entry) => `${entry.label}: ${entry.value}`),
	];
	const full = lines.join("\n");

	return full.length > MAX_TOTAL_CHARS
		? `${full.slice(0, MAX_TOTAL_CHARS)}…`
		: full;
}

export function usePageContext() {
	const capture = useCallback((): string => {
		const route = window.location.pathname;
		const semanticEntries = assistantContextBridge.snapshot().map((entry) => ({
			label: entry.label,
			key: entry.key,
			value:
				entry.value.length > MAX_CHARS
					? `${entry.value.slice(0, MAX_CHARS)}…`
					: entry.value,
		}));
		const domEntries = scrapeDomContext().filter(
			(entry) =>
				!semanticEntries.some((semantic) => semantic.key === entry.key),
		);

		return format(mergeEntries([...semanticEntries, ...domEntries]), route);
	}, []);

	return { capture };
}
