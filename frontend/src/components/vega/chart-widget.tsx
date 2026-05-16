"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";
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

Sizing: raw Vega specs need their width/height *signals* pushed when the
container resizes — `autosize.resize: true` only re-fits when the signals
change, and nothing changes them automatically. A ResizeObserver mirrors
the wrapper's measured size onto the Vega view, which causes the chart to
re-layout and re-fit to fill the panel. The Vega root SVG is centered via
flex and uses `display: block` to drop the inline baseline gap.
*/
export const ChartWidget = ({ spec, className }: ChartWidgetProps) => {
	const vegaContext = useVegaContext();
	const themeVersion = useThemeVersion();
	const viewRef = useRef<View | null>(null);
	const wrapperRef = useRef<HTMLDivElement | null>(null);

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

	const pushSize = useCallback(() => {
		const view = viewRef.current;
		const wrapper = wrapperRef.current;
		if (!view || !wrapper) return;
		const { width, height } = wrapper.getBoundingClientRect();
		if (width <= 0 || height <= 0) return;
		try {
			view
				.signal("width", Math.max(0, Math.floor(width)))
				.signal("height", Math.max(0, Math.floor(height)))
				.resize()
				.runAsync()
				.catch(() => {
					/* view may be torn down between resize fires; ignore */
				});
		} catch {
			/* signals are absent on some specs; ignore */
		}
	}, []);

	useEffect(() => {
		const wrapper = wrapperRef.current;
		if (!wrapper) return;
		const observer = new ResizeObserver(() => pushSize());
		observer.observe(wrapper);
		return () => observer.disconnect();
	}, [pushSize]);

	const handleEmbed = useCallback(
		(result: Result) => {
			viewRef.current = result.view;
			pushSize();
		},
		[pushSize],
	);

	return (
		<div
			className={cn(
				"flex h-full w-full items-center justify-center overflow-hidden [&_svg]:block",
				className,
			)}
			ref={wrapperRef}
		>
			<VegaEmbed
				className="flex h-full w-full items-center justify-center"
				onEmbed={handleEmbed}
				options={resolvedOptions}
				spec={resolvedSpec}
			/>
		</div>
	);
};
