"use client";

import { useMemo, useRef } from "react";
import { VegaEmbed } from "react-vega";
import type { View } from "vega";
import type { EmbedOptions, Result, VisualizationSpec } from "vega-embed";
import { cn } from "@/lib/utils";
import { useVegaContext } from "./context";
import { resolveSpecColors, useThemeVersion } from "./theme";

export interface ChartWidgetHandle {
	resize: (width: number, height: number) => void;
}

interface ChartWidgetProps {
	spec: VisualizationSpec;
	className?: string;
	ref?: React.Ref<ChartWidgetHandle>;
}

/*
ChartWidget renders any Vega or Vega-Lite spec. Before handing the spec to
vega-embed it walks every string and substitutes `var(--token)` references
with the live computed style value, and folds the provider's config in.
useThemeVersion invalidates both passes when the documentElement's class
or style attributes change (theme toggle), so colors stay in sync.
*/
export const ChartWidget = ({ spec, className }: ChartWidgetProps) => {
	const vegaContext = useVegaContext();
	const themeVersion = useThemeVersion();
	const viewRef = useRef<View | null>(null);

	const resolvedSpec = useMemo<VisualizationSpec>(() => {
		// themeVersion participates so the resolver re-runs on theme switch;
		// the resolver reads computed styles imperatively and never touches it.
		void themeVersion;
		const merged = {
			...(spec as Record<string, unknown>),
			config: {
				...(vegaContext.config ?? {}),
				...(((spec as Record<string, unknown>).config ?? {}) as Record<
					string,
					unknown
				>),
			},
		};
		return resolveSpecColors(merged) as VisualizationSpec;
	}, [spec, vegaContext.config, themeVersion]);

	const resolvedOptions = useMemo<EmbedOptions>(() => {
		void themeVersion;
		return resolveSpecColors(vegaContext.options ?? {}) as EmbedOptions;
	}, [vegaContext.options, themeVersion]);

	return (
		<div className={cn("h-full w-full", className)}>
			<VegaEmbed
				className="h-full w-full"
				onEmbed={(result: Result) => {
					viewRef.current = result.view;
				}}
				options={resolvedOptions}
				spec={resolvedSpec}
			/>
		</div>
	);
};
