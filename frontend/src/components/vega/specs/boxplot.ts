import type { Spec } from "./types";

interface BoxPlotSpecOptions {
	data: Array<{ category: string; value: number }>;
	categoryOrder?: string[];
	orientation?: "vertical" | "horizontal";
	categoryTitle?: string;
	valueTitle?: string;
	valueFormat?: string;
	extent?: "min-max" | number;
}

/*
boxPlotSpec renders one box-and-whisker per category. By default whiskers
extend to 1.5×IQR (Tukey) and outliers render as separate points. Pass
`extent: "min-max"` to stretch whiskers all the way to the data range, or
a numeric IQR multiplier for a custom rule. Orientation flips category to
the y-axis for long category names.
*/
export const boxPlotSpec = ({
	data,
	categoryOrder,
	orientation = "vertical",
	categoryTitle,
	valueTitle,
	valueFormat,
	extent = 1.5,
}: BoxPlotSpecOptions): Spec => {
	const categoryEnc = {
		axis: {
			domain: false,
			grid: false,
			labelPadding: 6,
			ticks: false,
			title: categoryTitle ?? null,
		},
		field: "category",
		scale: categoryOrder ? { domain: categoryOrder } : undefined,
		type: "nominal" as const,
	};

	const valueEnc = {
		axis: {
			domain: false,
			format: valueFormat,
			grid: true,
			gridDash: [2, 4],
			gridOpacity: 0.35,
			labelPadding: 6,
			tickCount: 4,
			ticks: false,
			title: valueTitle ?? null,
		},
		field: "value",
		scale: { nice: true, zero: false },
		type: "quantitative" as const,
	};

	const horizontal = orientation === "horizontal";

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		data: { values: data },
		encoding: {
			color: { field: "category", legend: null, type: "nominal" },
			x: horizontal ? valueEnc : categoryEnc,
			y: horizontal ? categoryEnc : valueEnc,
		},
		height: "container",
		mark: {
			box: { fillOpacity: 0.45, stroke: "var(--foreground)", strokeWidth: 1 },
			extent,
			median: { color: "var(--background)", strokeWidth: 1.5 },
			outliers: {
				fill: "var(--color-chart-2)",
				filled: true,
				size: 24,
				stroke: "var(--background)",
				strokeWidth: 0.75,
			},
			rule: { color: "var(--foreground)", strokeWidth: 1 },
			ticks: { color: "var(--foreground)" },
			type: "boxplot",
		},
		padding: { bottom: 4, left: 4, right: 8, top: 4 },
		width: "container",
	} as unknown as Spec;
};
