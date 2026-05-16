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
	normalize?: boolean;
	highlightDiagonal?: boolean;
}

/*
heatmapSpec renders a 2D rectangular heatmap. Confusion matrices are the
primary use case for benchmarks (rows = true class, cols = predicted), but
the same spec drives any (row, col) -> value visualization.

Visual choices: thin cell borders for separation, in-cell value labels
whose color flips against the background for legibility, an optional
row-normalized mode that shows recall instead of counts, and an optional
diagonal highlight so correct predictions read at a glance.
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
	normalize = false,
	highlightDiagonal = false,
}: HeatmapSpecOptions): Spec => {
	const rowTotals = new Map<string, number>();
	for (const cell of data) {
		rowTotals.set(cell.row, (rowTotals.get(cell.row) ?? 0) + cell.value);
	}

	const annotated = data.map((cell) => {
		const total = rowTotals.get(cell.row) ?? 0;
		const display = normalize && total > 0 ? cell.value / total : cell.value;
		return {
			...cell,
			diagonal: cell.row === cell.col ? 1 : 0,
			display,
			rowTotal: total,
		};
	});

	const maxDisplay = annotated.reduce(
		(acc, cell) => (cell.display > acc ? cell.display : acc),
		0,
	);

	const layers: Record<string, unknown>[] = [
		{
			encoding: {
				color: {
					field: "display",
					legend: {
						gradientLength: 100,
						gradientThickness: 8,
						title: null,
					},
					scale: { scheme, zero: true },
					type: "quantitative",
				},
				stroke: { value: "var(--background)" },
				strokeWidth: { value: 1 },
				tooltip: [
					{ field: "row", title: rowTitle, type: "nominal" },
					{ field: "col", title: colTitle, type: "nominal" },
					{
						field: "value",
						format: "d",
						title: "Count",
						type: "quantitative",
					},
					{
						field: "display",
						format: normalize ? ".1%" : "d",
						title: normalize ? "Row %" : "Value",
						type: "quantitative",
					},
				],
			},
			mark: { type: "rect" },
		},
	];

	if (highlightDiagonal) {
		layers.push({
			encoding: {
				fillOpacity: { value: 0 },
				stroke: {
					condition: {
						test: "datum.row === datum.col",
						value: "var(--color-chart-1)",
					},
					value: "transparent",
				},
				strokeWidth: {
					condition: { test: "datum.row === datum.col", value: 1.5 },
					value: 0,
				},
			},
			mark: { type: "rect" },
		});
	}

	if (showValues) {
		layers.push({
			encoding: {
				color: {
					condition: {
						test: `datum.display > ${maxDisplay / 2}`,
						value: "white",
					},
					value: "var(--foreground)",
				},
				text: {
					field: "display",
					format: normalize ? ".0%" : valueFormat,
					type: "quantitative",
				},
			},
			mark: { fontSize: 11, fontWeight: 500, type: "text" },
		});
	}

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		data: { values: annotated },
		encoding: {
			x: {
				axis: {
					domain: false,
					labelAngle: 0,
					labelPadding: 6,
					ticks: false,
					title: colTitle,
				},
				field: "col",
				scale: colOrder
					? { domain: colOrder, paddingInner: 0.04 }
					: { paddingInner: 0.04 },
				sort: colOrder,
				type: "nominal",
			},
			y: {
				axis: {
					domain: false,
					labelPadding: 6,
					ticks: false,
					title: rowTitle,
				},
				field: "row",
				scale: rowOrder
					? { domain: rowOrder, paddingInner: 0.04 }
					: { paddingInner: 0.04 },
				sort: rowOrder,
				type: "nominal",
			},
		},
		height: "container",
		layer: layers,
		width: "container",
	} as unknown as Spec;
};
