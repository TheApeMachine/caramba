import type { Spec } from "./types";

export interface BarDatum {
	label: string;
	value: number;
}

interface BarSpecOptions {
	data: BarDatum[];
	orientation?: "vertical" | "horizontal";
}

/*
barSpec produces a simple bar chart. Orientation switches the axis bindings
without changing the data shape — same factory powers both bar-vertical and
bar-horizontal use cases.
*/
export const barSpec = ({
	data,
	orientation = "vertical",
}: BarSpecOptions): Spec => {
	const isVertical = orientation === "vertical";

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		data: { values: data },
		background: "transparent",
		encoding: {
			tooltip: [
				{ field: "label", title: "Label", type: "nominal" },
				{ field: "value", title: "Value", type: "quantitative" },
			],
			[isVertical ? "x" : "y"]: {
				axis: { title: null },
				field: "label",
				type: "nominal",
			},
			[isVertical ? "y" : "x"]: {
				axis: { grid: false, title: null },
				field: "value",
				type: "quantitative",
			},
		},
		height: "container",
		mark: { type: "bar" },
		width: "container",
	} as Spec;
};

interface StackedBarOptions {
	data: Array<Record<string, unknown> & { name: string }>;
	seriesKeys: string[];
	orientation?: "vertical" | "horizontal";
	normalize?: boolean;
}

/*
stackedBarSpec covers both the stacked-vertical and stacked-horizontal cases
plus normalized (percentage) stacking, gating one factory across the previous
three components.
*/
export const stackedBarSpec = ({
	data,
	seriesKeys,
	orientation = "vertical",
	normalize = false,
}: StackedBarOptions): Spec => {
	const isVertical = orientation === "vertical";
	const vegaData = data.flatMap((row, idx) =>
		seriesKeys.map((key) => ({
			category: row.name ?? `Row ${idx}`,
			series: key,
			value: typeof row[key] === "number" ? (row[key] as number) : 0,
		})),
	);

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		data: { values: vegaData },
		encoding: {
			color: {
				field: "series",
				legend: { orient: "top" },
				type: "nominal",
			},
			tooltip: [
				{ field: "category", title: "Category", type: "nominal" },
				{ field: "series", title: "Series", type: "nominal" },
				{
					field: "value",
					format: ".1f",
					title: "Value",
					type: "quantitative",
				},
			],
			[isVertical ? "x" : "y"]: {
				axis: { title: null },
				field: "category",
				type: "nominal",
			},
			[isVertical ? "y" : "x"]: {
				axis: { title: null },
				field: "value",
				stack: normalize ? "normalize" : "zero",
				type: "quantitative",
			},
		},
		height: "container",
		mark: { opacity: 0.85, type: "bar" },
		width: "container",
	} as Spec;
};
