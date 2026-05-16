import type { Spec } from "./types";

interface LineSpecOptions {
	data: Array<Record<string, number | string>>;
	xField: string;
	seriesKeys: string[];
	seriesLabels?: Record<string, string>;
	xTitle?: string;
	yTitle?: string;
	xType?: "quantitative" | "temporal";
	yDomain?: [number | null, number | null];
	yFormat?: string;
	smooth?: boolean;
}

/*
lineSpec is a multi-series line chart on a continuous x-axis (training steps,
elapsed seconds, anything quantitative). The dashboard's areaSpec assumes a
temporal axis with dates; benchmarks usually plot against step counters, so
this stays separate rather than overloading areaSpec with axis-type branching.
*/
export const lineSpec = ({
	data,
	xField,
	seriesKeys,
	seriesLabels,
	xTitle,
	yTitle,
	xType = "quantitative",
	yDomain = [0, null],
	yFormat,
	smooth = true,
}: LineSpecOptions): Spec => {
	const vegaData = data.flatMap((row) =>
		seriesKeys.map((key) => ({
			x: row[xField],
			series: seriesLabels?.[key] ?? key,
			value: typeof row[key] === "number" ? (row[key] as number) : 0,
		})),
	);

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		data: { values: vegaData },
		encoding: {
			color: {
				field: "series",
				legend: { orient: "top", title: null },
				type: "nominal",
			},
			tooltip: [
				{ field: "x", title: xTitle ?? "x", type: xType },
				{ field: "series", title: "Series", type: "nominal" },
				{
					field: "value",
					format: yFormat ?? ".3f",
					title: yTitle ?? "Value",
					type: "quantitative",
				},
			],
			x: {
				axis: { grid: false, title: xTitle ?? null },
				field: "x",
				type: xType,
			},
			y: {
				axis: { format: yFormat, title: yTitle ?? null },
				field: "value",
				scale: { domain: yDomain },
				type: "quantitative",
			},
		},
		height: "container",
		mark: {
			interpolate: smooth ? "monotone" : "linear",
			point: false,
			strokeWidth: 2,
			type: "line",
		},
		width: "container",
	} as Spec;
};
