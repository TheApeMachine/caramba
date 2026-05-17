/*
usePageContext scrapes data-context elements from the live DOM on demand.

Opt any element in by adding:
  data-context="<label>"          — human-readable label for this chunk
  data-context-key="<key>"        — optional machine key (defaults to label)
  data-context-type="text|value|json|count"
                                  — how to extract (default: "text")

The hook returns a `capture()` function that collects a snapshot and formats
it as a compact string suitable for prepending to an agent message as context.

Rules that keep context size reasonable:
- Only elements with data-context are included (explicit opt-in).
- Each extracted value is truncated at MAX_CHARS characters.
- The full snapshot is capped at MAX_TOTAL_CHARS.
- Elements hidden via display:none or visibility:hidden are skipped.
*/

const MAX_CHARS = 512;
const MAX_TOTAL_CHARS = 4096;

type ContextEntry = {
	label: string;
	key: string;
	value: string;
};

type ExtractionType = "text" | "value" | "json" | "count";

function isVisible(el: Element): boolean {
	const style = window.getComputedStyle(el);
	return (
		style.display !== "none" &&
		style.visibility !== "hidden" &&
		style.opacity !== "0"
	);
}

function extractValue(el: Element, type: ExtractionType): string {
	switch (type) {
		case "value": {
			const input = el as
				| HTMLInputElement
				| HTMLTextAreaElement
				| HTMLSelectElement;
			return input.value ?? "";
		}
		case "json": {
			const raw = el.getAttribute("data-context-value") ?? el.textContent ?? "";
			try {
				return JSON.stringify(JSON.parse(raw), null, 2);
			} catch {
				return raw;
			}
		}
		case "count": {
			const children = el.querySelectorAll("[data-context-item]");
			return children.length > 0
				? `${children.length} item${children.length === 1 ? "" : "s"}`
				: el.childElementCount.toString();
		}
		default: {
			return (el.textContent ?? "").replace(/\s+/g, " ").trim();
		}
	}
}

function scrape(): ContextEntry[] {
	const nodes = document.querySelectorAll("[data-context]");
	const entries: ContextEntry[] = [];

	for (const node of nodes) {
		if (!isVisible(node)) continue;

		const label = node.getAttribute("data-context") ?? "";
		const key =
			node.getAttribute("data-context-key") ??
			label.toLowerCase().replace(/\s+/g, "_");
		const type = (node.getAttribute("data-context-type") ??
			"text") as ExtractionType;

		const raw = extractValue(node, type);
		const value = raw.length > MAX_CHARS ? `${raw.slice(0, MAX_CHARS)}…` : raw;

		if (value) entries.push({ label, key, value });
	}

	return entries;
}

function format(entries: ContextEntry[], route: string): string {
	if (entries.length === 0) return "";

	const lines: string[] = [
		`[Page context — ${route}]`,
		...entries.map((e) => `${e.label}: ${e.value}`),
	];

	const full = lines.join("\n");
	return full.length > MAX_TOTAL_CHARS
		? `${full.slice(0, MAX_TOTAL_CHARS)}…`
		: full;
}

export function usePageContext() {
	const capture = (): string => {
		const route = window.location.pathname;
		const entries = scrape();
		return format(entries, route);
	};

	return { capture };
}
