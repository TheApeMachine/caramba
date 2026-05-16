import type { Spec } from "./types";

interface HistogramSpecOptions {
	values: number[];
	bins?: number;
	xTitle?: string;
	yTitle?: string;
	xFormat?: string;
	highlightQuantiles?: boolean;
}

/*
histogramSpec bins a flat array and renders the distribution as bars.
Visual choices: rounded top corners, a subtle hover state via opacity
selection, and optional p50/p95 reference rules so a benchmark reviewer
can see tail behavior at a glance.
*/
export const histogramSpec = ({
	values,
	bins = 30,
	xTitle,
	yTitle = "count",
	xFormat,
	highlightQuantiles = true,
}: HistogramSpecOptions): Spec => {
	const layers: Record<string, unknown>[] = [
		{
			encoding: {
				color: { value: "var(--color-chart-1)" },
				opacity: {
					condition: { empty: false, param: "barHover", value: 1 },
					value: 0.78,
				},
				tooltip: [
					{
						bin: { maxbins: bins },
						field: "value",
						format: xFormat ?? ".2f",
						title: xTitle ?? "Bin",
						type: "quantitative",
					},
					{ aggregate: "count", title: yTitle, type: "quantitative" },
				],
				x: {
					axis: {
						domain: false,
						format: xFormat,
						grid: false,
						labelPadding: 6,
						ticks: false,
						title: xTitle ?? null,
					},
					bin: { maxbins: bins },
					field: "value",
					type: "quantitative",
				},
				x2: { bin: { maxbins: bins }, field: "value" },
				y: {
					aggregate: "count",
					axis: {
						domain: false,
						grid: true,
						gridDash: [2, 4],
						gridOpacity: 0.35,
						labelPadding: 6,
						tickCount: 3,
						ticks: false,
						title: yTitle,
					},
					type: "quantitative",
				},
			},
			mark: {
				cornerRadiusTopLeft: 3,
				cornerRadiusTopRight: 3,
				type: "bar",
			},
			params: [
				{
					name: "barHover",
					select: {
						clear: "pointerout",
						on: "pointerover",
						type: "point",
					},
				},
			],
		},
	];

	if (highlightQuantiles && values.length >= 4) {
		const sorted = [...values].sort((a, b) => a - b);
		const q = (p: number) =>
			sorted[Math.min(sorted.length - 1, Math.floor(p * (sorted.length - 1)))];
		const refs = [
			{ label: "p50", value: q(0.5) },
			{ label: "p95", value: q(0.95) },
		];

		layers.push({
			data: { values: refs },
			encoding: {
				color: { value: "var(--muted-foreground)" },
				strokeDash: { value: [3, 3] },
				tooltip: [
					{ field: "label", title: "Quantile", type: "nominal" },
					{
						field: "value",
						format: xFormat ?? ".2f",
						title: xTitle ?? "Value",
						type: "quantitative",
					},
				],
				x: { field: "value", type: "quantitative" },
			},
			mark: { strokeWidth: 1, type: "rule" },
		});

		layers.push({
			data: { values: refs },
			encoding: {
				color: { value: "var(--muted-foreground)" },
				text: { field: "label", type: "nominal" },
				x: { field: "value", type: "quantitative" },
			},
			mark: {
				align: "left",
				baseline: "top",
				dx: 4,
				dy: 4,
				fontSize: 10,
				type: "text",
				y: 4,
			},
		});
	}

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		data: { values: values.map((value) => ({ value })) },
		height: "container",
		layer: layers,
		padding: { bottom: 4, left: 4, right: 4, top: 4 },
		width: "container",
	} as unknown as Spec;
};
