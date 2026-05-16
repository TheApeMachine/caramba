"use client";

import { createContext, useContext, useMemo } from "react";
import type { EmbedOptions } from "vega-embed";

/*
VegaContextValue holds shared rendering defaults for every chart in a tree.
Specs stay pure data; the provider supplies theme, renderer, formats, and
shared signals so widgets don't each re-declare boilerplate.
*/
export interface VegaContextValue {
	config: Record<string, unknown>;
	renderer: "canvas" | "svg";
	options: EmbedOptions;
	scheme?: string;
	signals?: Record<string, unknown>;
}

/*
defaultVegaConfig mirrors the styling previously hard-coded across each chart
file (axis colors, legend colors, transparent view stroke). Defining it once
here means a single edit retones every widget.
*/
export const defaultVegaConfig: Record<string, unknown> = {
	axis: {
		domainColor: "oklch(var(--muted-foreground))",
		gridColor: "oklch(var(--border))",
		labelColor: "oklch(var(--foreground))",
		tickColor: "oklch(var(--muted-foreground))",
		titleColor: "oklch(var(--foreground))",
	},
	legend: {
		labelColor: "oklch(var(--foreground))",
		titleColor: "oklch(var(--foreground))",
	},
	range: {
		category: [
			"oklch(var(--chart-1))",
			"oklch(var(--chart-2))",
			"oklch(var(--chart-3))",
			"oklch(var(--chart-4))",
			"oklch(var(--chart-5))",
		],
	},
	view: { stroke: "transparent" },
};

const VegaContext = createContext<VegaContextValue>({
	config: defaultVegaConfig,
	options: { actions: false, renderer: "canvas" },
	renderer: "canvas",
});

interface VegaProviderProps {
	value?: Partial<VegaContextValue>;
	children: React.ReactNode;
}

/*
VegaProvider supplies merged defaults to every descendant chart. Callers can
override any field (renderer for SSR/export, config for a different theme,
options for showing the actions menu) without re-implementing the rest.
*/
export const VegaProvider = ({ value, children }: VegaProviderProps) => {
	const merged = useMemo<VegaContextValue>(
		() => ({
			config: { ...defaultVegaConfig, ...(value?.config ?? {}) },
			options: {
				actions: false,
				renderer: value?.renderer ?? "canvas",
				...(value?.options ?? {}),
			},
			renderer: value?.renderer ?? "canvas",
			scheme: value?.scheme,
			signals: value?.signals,
		}),
		[value],
	);

	return <VegaContext.Provider value={merged}>{children}</VegaContext.Provider>;
};

/*
useVegaContext returns the active Vega defaults. Always wrapped in a default
value so widgets work without an explicit provider.
*/
export const useVegaContext = () => useContext(VegaContext);
