import type { Spec } from "./types";

export interface BarDatum {
	label: string;
	value: number;
}

interface BarSpecOptions {
	data: BarDatum[];
	orientation?: "vertical" | "horizontal";
	valueFormat?: string;
	axisTitle?: string;
}

const colorEnc = {
	field: "series",
	legend: null,
	type: "nominal" as const,
};

/*
barSpec is the plain single-series bar. Rounded top corners, hover dim on
non-hovered bars, theme palette via category scale. Orientation flips the
axis bindings without altering data shape.
*/
export const barSpec = ({
	data,
	orientation = "vertical",
	valueFormat,
	axisTitle,
}: BarSpecOptions): Spec => {
	const horizontal = orientation === "horizontal";
	const tagged = data.map((datum) => ({ ...datum, series: "value" }));

	const valueEnc = {
		axis: {
			domain: false,
			format: valueFormat,
			grid: horizontal,
			gridDash: [2, 4],
			gridOpacity: 0.35,
			labelPadding: 6,
			tickCount: 4,
			ticks: false,
			title: axisTitle ?? null,
		},
		field: "value",
		scale: { nice: true, zero: true },
		type: "quantitative" as const,
	};

	const categoryEnc = {
		axis: { domain: false, labelPadding: 6, ticks: false, title: null },
		field: "label",
		scale: { paddingInner: 0.25 },
		sort: null,
		type: "nominal" as const,
	};

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		data: { values: tagged },
		encoding: {
			color: colorEnc,
			opacity: {
				condition: { empty: false, param: "barHover", value: 1 },
				value: 0.85,
			},
			tooltip: [
				{ field: "label", title: "Label", type: "nominal" },
				{
					field: "value",
					format: valueFormat ?? ".2f",
					title: "Value",
					type: "quantitative",
				},
			],
			[horizontal ? "x" : "y"]: valueEnc,
			[horizontal ? "y" : "x"]: categoryEnc,
		},
		height: "container",
		mark: {
			cornerRadiusBottomRight: horizontal ? 3 : 0,
			cornerRadiusTopLeft: horizontal ? 0 : 3,
			cornerRadiusTopRight: 3,
			type: "bar",
		},
		padding: { bottom: 4, left: 4, right: 4, top: 4 },
		params: [
			{
				name: "barHover",
				select: { clear: "pointerout", on: "pointerover", type: "point" },
			},
		],
		width: "container",
	} as unknown as Spec;
};

interface StackedBarOptions {
	data: Array<Record<string, unknown> & { name: string }>;
	seriesKeys: string[];
	orientation?: "vertical" | "horizontal";
	normalize?: boolean;
	valueFormat?: string;
}

/*
stackedBarSpec stacks multiple series per category. Top corners on the
final segment of each stack are rounded; intermediate segments stay
square so the stack reads as one shape. Normalize switches the value
axis to percent.
*/
export const stackedBarSpec = ({
	data,
	seriesKeys,
	orientation = "vertical",
	normalize = false,
	valueFormat,
}: StackedBarOptions): Spec => {
	const horizontal = orientation === "horizontal";
	const vegaData = data.flatMap((row, idx) =>
		seriesKeys.map((key) => ({
			category: row.name ?? `Row ${idx}`,
			series: key,
			value: typeof row[key] === "number" ? (row[key] as number) : 0,
		})),
	);

	const valueFmt = normalize ? ".0%" : (valueFormat ?? ".0f");

	const valueEnc = {
		axis: {
			domain: false,
			format: valueFmt,
			grid: true,
			gridDash: [2, 4],
			gridOpacity: 0.35,
			labelPadding: 6,
			tickCount: 4,
			ticks: false,
			title: null,
		},
		field: "value",
		scale: { nice: true, zero: true },
		stack: normalize ? ("normalize" as const) : ("zero" as const),
		type: "quantitative" as const,
	};

	const categoryEnc = {
		axis: { domain: false, labelPadding: 6, ticks: false, title: null },
		field: "category",
		scale: { paddingInner: 0.3 },
		type: "nominal" as const,
	};

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		data: { values: vegaData },
		encoding: {
			color: {
				field: "series",
				legend: {
					offset: 4,
					orient: "top",
					symbolType: "square",
					title: null,
				},
				scale: { domain: seriesKeys },
				type: "nominal",
			},
			order: { field: "series", sort: "ascending", type: "nominal" },
			tooltip: [
				{ field: "category", title: "Category", type: "nominal" },
				{ field: "series", title: "Series", type: "nominal" },
				{
					field: "value",
					format: valueFmt,
					title: "Value",
					type: "quantitative",
				},
			],
			[horizontal ? "x" : "y"]: valueEnc,
			[horizontal ? "y" : "x"]: categoryEnc,
		},
		height: "container",
		mark: { type: "bar" },
		padding: { bottom: 4, left: 4, right: 4, top: 4 },
		width: "container",
	} as unknown as Spec;
};
