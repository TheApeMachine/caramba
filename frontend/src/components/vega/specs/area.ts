import type { SeriesPoint, Spec } from "./types";

interface AreaSpecOptions {
	data: SeriesPoint[];
	seriesKeys: string[];
	seriesLabels?: Record<string, string>;
}

/*
areaSpec produces a multi-series area chart over time, with color encoding
driven by the series name. Color palette comes from the active VegaContext
(config.range.category), so no palette is baked into the spec.
*/
export const areaSpec = ({
	data,
	seriesKeys,
	seriesLabels,
}: AreaSpecOptions): Spec => {
	const vegaData = data.flatMap((point) =>
		seriesKeys.map((key) => ({
			date: point.date,
			series: seriesLabels?.[key] ?? key,
			value: point[key] ?? 0,
		})),
	);

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		data: { values: vegaData },
		encoding: {
			background: "transparent",
			color: { field: "series", legend: null, type: "nominal" },
			tooltip: [
				{
					field: "date",
					format: "%b %d, %Y",
					title: "Date",
					type: "temporal",
				},
				{ field: "series", title: "Series", type: "nominal" },
				{ field: "value", title: "Value", type: "quantitative" },
			],
			x: {
				axis: { format: "%b %d", grid: false, title: null },
				field: "date",
				type: "temporal",
			},
			y: {
				axis: { grid: false, title: null },
				field: "value",
				scale: { domain: [0, null] },
				type: "quantitative",
			},
		},
		height: "container",
		mark: { line: true, opacity: 0.3, type: "area" },
		width: "container",
	} as Spec;
};
