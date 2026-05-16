"use client";

import { useSyncExternalStore } from "react";

/*
Vega parses every color through d3-color before it reaches the renderer.
"var(--foo)" is not a parseable color, so Vega silently falls back to its
defaults (typically black). The fix is to resolve CSS variables to their
actual computed values in JS and substitute them into the spec before it
hits Vega.

resolveSpecColors walks a spec and replaces every "var(--xxx)" reference
inside any string value with the resolved color. The spec is treated as
plain JSON; we only allocate new arrays/objects on the path where a
substitution actually happened, so unchanged subtrees are shared.

useThemeVersion subscribes to mutations of the documentElement's classList
(that's where theme classes "dark"/"dim" toggle) and bumps a counter, so
specs can be re-resolved on theme switch via a useMemo dependency.
*/

const VAR_PATTERN = /var\(\s*(--[\w-]+)\s*(?:,[^)]*)?\)/g;

const cache = new Map<string, string>();
let cachedThemeKey: string | null = null;

const themeKey = (): string => {
	if (typeof document === "undefined") return "";
	return document.documentElement.className;
};

const resolveVar = (name: string): string => {
	if (typeof document === "undefined") return "";
	const key = themeKey();
	if (key !== cachedThemeKey) {
		cache.clear();
		cachedThemeKey = key;
	}
	const hit = cache.get(name);
	if (hit !== undefined) return hit;
	const value = getComputedStyle(document.documentElement)
		.getPropertyValue(name)
		.trim();
	cache.set(name, value);
	return value;
};

const resolveString = (input: string): string => {
	if (!input.includes("var(")) return input;
	let mutated = false;
	const out = input.replace(VAR_PATTERN, (whole, name) => {
		const resolved = resolveVar(name);
		if (!resolved) return whole;
		mutated = true;
		return resolved;
	});
	return mutated ? out : input;
};

export const resolveSpecColors = <T>(spec: T): T => {
	if (typeof spec === "string") {
		const resolved = resolveString(spec);
		return (resolved === spec ? spec : resolved) as unknown as T;
	}

	if (Array.isArray(spec)) {
		let changed = false;
		const out = spec.map((entry) => {
			const next = resolveSpecColors(entry);
			if (next !== entry) changed = true;
			return next;
		});
		return (changed ? out : spec) as unknown as T;
	}

	if (spec && typeof spec === "object") {
		const source = spec as Record<string, unknown>;
		let changed = false;
		const out: Record<string, unknown> = {};
		for (const key of Object.keys(source)) {
			const next = resolveSpecColors(source[key]);
			if (next !== source[key]) changed = true;
			out[key] = next;
		}
		return (changed ? out : spec) as unknown as T;
	}

	return spec;
};

const subscribers = new Set<() => void>();
let observer: MutationObserver | null = null;

const ensureObserver = () => {
	if (observer || typeof document === "undefined") return;
	observer = new MutationObserver(() => {
		cache.clear();
		cachedThemeKey = null;
		for (const listener of subscribers) listener();
	});
	observer.observe(document.documentElement, {
		attributeFilter: ["class", "style"],
		attributes: true,
	});
};

const subscribe = (listener: () => void): (() => void) => {
	ensureObserver();
	subscribers.add(listener);
	return () => {
		subscribers.delete(listener);
	};
};

const getSnapshot = () => themeKey();
const getServerSnapshot = () => "";

export const useThemeVersion = (): string =>
	useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
