import type { Spec } from "./types";

interface HeatmapCell {
	row: string;
	col: string;
	value: number;
}

interface HeatmapSpecOptions {
	data: HeatmapCell[];
	rowOrder?: string[];
	colOrder?: string[];
	rowTitle?: string;
	colTitle?: string;
	valueFormat?: string;
	showValues?: boolean;
	scheme?: string;
}

/*
heatmapSpec renders a 2D rectangular heatmap. Confusion matrices are the
primary use case for benchmarks (rows = true class, cols = predicted), but
the same spec drives any (row, col) -> value visualization. valueFormat
controls the cell label format when showValues is true.
*/
export const heatmapSpec = ({
	data,
	rowOrder,
	colOrder,
	rowTitle = "Actual",
	colTitle = "Predicted",
	valueFormat = "d",
	showValues = true,
	scheme = "blues",
}: HeatmapSpecOptions): Spec => {
	const layers: Record<string, unknown>[] = [
		{
			encoding: {
				color: {
					field: "value",
					legend: { title: null },
					scale: { scheme },
					type: "quantitative",
				},
				tooltip: [
					{ field: "row", title: rowTitle, type: "nominal" },
					{ field: "col", title: colTitle, type: "nominal" },
					{
						field: "value",
						format: valueFormat,
						title: "Value",
						type: "quantitative",
					},
				],
			},
			mark: { type: "rect" },
		},
	];

	if (showValues) {
		layers.push({
			encoding: {
				color: {
					condition: {
						test: "datum.value > (datum.__max / 2)",
						value: "white",
					},
					value: "oklch(var(--foreground))",
				},
				text: { field: "value", format: valueFormat, type: "quantitative" },
			},
			mark: { fontSize: 11, type: "text" },
		});
	}

	const maxValue = data.reduce(
		(acc, cell) => (cell.value > acc ? cell.value : acc),
		0,
	);
	const annotated = data.map((cell) => ({ ...cell, __max: maxValue }));

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		data: { values: annotated },
		encoding: {
			x: {
				axis: { labelAngle: 0, title: colTitle },
				field: "col",
				scale: colOrder ? { domain: colOrder } : undefined,
				sort: colOrder,
				type: "nominal",
			},
			y: {
				axis: { title: rowTitle },
				field: "row",
				scale: rowOrder ? { domain: rowOrder } : undefined,
				sort: rowOrder,
				type: "nominal",
			},
		},
		height: "container",
		layer: layers,
		width: "container",
	} as Spec;
};
