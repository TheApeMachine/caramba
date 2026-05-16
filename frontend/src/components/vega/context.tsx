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
		domainColor: "var(--muted-foreground)",
		gridColor: "var(--border)",
		labelColor: "var(--foreground)",
		labelFont: "Inter, sans-serif",
		labelFontSize: 10,
		tickColor: "var(--muted-foreground)",
		titleColor: "var(--muted-foreground)",
		titleFont: "Inter, sans-serif",
		titleFontSize: 10,
		titleFontWeight: 500,
	},
	legend: {
		labelColor: "var(--foreground)",
		labelFont: "Inter, sans-serif",
		labelFontSize: 11,
		symbolStrokeWidth: 2,
		titleColor: "var(--muted-foreground)",
		titleFont: "Inter, sans-serif",
		titleFontSize: 10,
	},
	range: {
		category: [
			"var(--color-chart-1)",
			"var(--color-chart-2)",
			"var(--color-chart-3)",
			"var(--color-chart-4)",
			"var(--color-chart-5)",
			"var(--color-chart-6)",
			"var(--color-chart-7)",
			"var(--color-chart-8)",
			"var(--color-chart-9)",
			"var(--color-chart-10)",
		],
		heatmap: { scheme: "blues" },
		ramp: { scheme: "blues" },
	},
	style: {
		"guide-label": { fill: "var(--foreground)" },
		"guide-title": { fill: "var(--muted-foreground)" },
	},
	title: {
		color: "var(--foreground)",
		font: "Inter, sans-serif",
		subtitleColor: "var(--muted-foreground)",
	},
	view: { stroke: "transparent" },
};

const VegaContext = createContext<VegaContextValue>({
	config: defaultVegaConfig,
	options: { actions: false, renderer: "svg" },
	renderer: "svg",
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
				renderer: value?.renderer ?? "svg",
				...(value?.options ?? {}),
			},
			renderer: value?.renderer ?? "svg",
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
