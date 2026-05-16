import type { Spec } from "./types";

interface HeatmapTrace {
	points: Array<{ x: number; y: number }>;
	color?: string;
	strokeWidth?: number;
	dash?: boolean;
	label?: string;
}

interface DenseHeatmapOptions {
	matrix: number[][];
	xTitle?: string;
	yTitle?: string;
	valueTitle?: string;
	valueFormat?: string;
	scheme?: string;
	logScale?: boolean;
	xExtent?: [number, number];
	yExtent?: [number, number];
	traces?: HeatmapTrace[];
}

/*
denseHeatmapSpec renders a 2D matrix as an image-style heatmap with
continuous numeric axes — the right primitive for similarity matrices,
spectrograms, token-frequency maps, and attention patterns. Caller passes
a row-major `number[][]` where matrix[y][x] is the value; row 0 is the
bottom of the chart so increasing y points up. xExtent/yExtent let the
caller map matrix indices to real units (seconds, Hz, position).

Optional `traces` overlay one or more polylines on top of the heatmap —
the spectrogram pattern where a tracked frequency or ridge is drawn over
the density background.
*/
export const denseHeatmapSpec = ({
	matrix,
	xTitle,
	yTitle,
	valueTitle = "value",
	valueFormat = ".3f",
	scheme = "inferno",
	logScale = false,
	xExtent,
	yExtent,
	traces = [],
}: DenseHeatmapOptions): Spec => {
	const rows = matrix.length;
	const cols = rows > 0 ? matrix[0].length : 0;

	if (rows === 0 || cols === 0) {
		return {
			$schema: "https://vega.github.io/schema/vega-lite/v6.json",
			autosize: { contains: "padding", resize: true, type: "fit" },
			background: "transparent",
			data: { values: [] },
			height: "container",
			mark: "rect",
			width: "container",
		} as unknown as Spec;
	}

	const [x0, x1] = xExtent ?? [0, cols];
	const [y0, y1] = yExtent ?? [0, rows];
	const dx = (x1 - x0) / cols;
	const dy = (y1 - y0) / rows;

	const cells: Array<{
		x: number;
		y: number;
		x2: number;
		y2: number;
		value: number;
	}> = [];

	for (let yi = 0; yi < rows; yi++) {
		for (let xi = 0; xi < cols; xi++) {
			const raw = matrix[yi][xi];
			cells.push({
				value: logScale ? Math.log(raw + 1) : raw,
				x: x0 + xi * dx,
				x2: x0 + (xi + 1) * dx,
				y: y0 + yi * dy,
				y2: y0 + (yi + 1) * dy,
			});
		}
	}

	const xEnc = {
		axis: {
			domain: false,
			grid: false,
			labelPadding: 6,
			ticks: false,
			title: xTitle ?? null,
		},
		field: "x",
		scale: { domain: [x0, x1], nice: false, zero: false },
		type: "quantitative" as const,
	};

	const yEnc = {
		axis: {
			domain: false,
			grid: false,
			labelPadding: 6,
			ticks: false,
			title: yTitle ?? null,
		},
		field: "y",
		scale: { domain: [y0, y1], nice: false, zero: false },
		type: "quantitative" as const,
	};

	const layers: Record<string, unknown>[] = [
		{
			data: { values: cells },
			encoding: {
				color: {
					field: "value",
					legend: {
						gradientLength: { expr: "max(40, height - 20)" },
						gradientThickness: 8,
						title: logScale ? `log(${valueTitle} + 1)` : valueTitle,
					},
					scale: { scheme },
					type: "quantitative",
				},
				tooltip: [
					{
						field: "x",
						format: ".2f",
						title: xTitle ?? "x",
						type: "quantitative",
					},
					{
						field: "y",
						format: ".2f",
						title: yTitle ?? "y",
						type: "quantitative",
					},
					{
						field: "value",
						format: valueFormat,
						title: valueTitle,
						type: "quantitative",
					},
				],
				x: xEnc,
				x2: { field: "x2" },
				y: yEnc,
				y2: { field: "y2" },
			},
			mark: { type: "rect" },
		},
	];

	traces.forEach((trace, traceIdx) => {
		const indexed = trace.points.map((point, order) => ({
			...point,
			order,
			traceIdx,
		}));

		layers.push({
			data: { values: indexed },
			encoding: {
				color: { value: trace.color ?? "var(--color-chart-2)" },
				order: { field: "order", type: "quantitative" },
				strokeDash: { value: trace.dash ? [4, 3] : [1, 0] },
				x: xEnc,
				y: yEnc,
			},
			mark: {
				interpolate: "linear",
				strokeCap: "round",
				strokeJoin: "round",
				strokeWidth: trace.strokeWidth ?? 1.5,
				type: "line",
			},
		});
	});

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		height: "container",
		layer: layers,
		padding: { bottom: 4, left: 4, right: 8, top: 4 },
		width: "container",
	} as unknown as Spec;
};
