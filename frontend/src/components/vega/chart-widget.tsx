"use client";

import { useRef } from "react";
import { VegaEmbed } from "react-vega";
import type { View } from "vega";
import type { Result, VisualizationSpec } from "vega-embed";
import { cn } from "@/lib/utils";
import { useVegaContext } from "./context";

export interface ChartWidgetHandle {
	resize: (width: number, height: number) => void;
}

interface ChartWidgetProps {
	spec: VisualizationSpec;
	className?: string;
	ref?: React.Ref<ChartWidgetHandle>;
}

/*
ChartWidget is the single uniform widget that renders any Vega or Vega-Litespec.
*/
export const ChartWidget = ({ spec, className }: ChartWidgetProps) => {
	const vegaContext = useVegaContext();
	const viewRef = useRef<View | null>(null);

	return (
		<div className={cn("h-full w-full", className)}>
			<VegaEmbed
				spec={spec}
				options={vegaContext.options}
				onEmbed={(result: Result) => {
					viewRef.current = result.view;
				}}
				className="h-full w-full"
			/>
		</div>
	);
};
