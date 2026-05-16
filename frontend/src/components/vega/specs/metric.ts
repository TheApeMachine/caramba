import type { Spec } from "./types";

interface MetricSpecOptions {
	label: string;
	value: number;
}

/*
metricSpec uses a Vega-Lite text mark to render a "big number" card with a
small caption underneath. Keeping the metric inside the spec contract means
the dashboard treats it identically to any other widget — no special case.
*/
export const metricSpec = ({ label, value }: MetricSpecOptions): Spec =>
	({
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		config: { view: { stroke: "transparent" } },
		data: {
			values: [
				{ kind: "value", text: value.toLocaleString(), y: 0.4 },
				{ kind: "label", text: label, y: 0.7 },
			],
		},
		encoding: {
			text: { field: "text", type: "nominal" },
			x: { value: { expr: "width / 2" } },
			y: { field: "y", scale: { domain: [0, 1] }, type: "quantitative" },
		},
		height: "container",
		layer: [
			{
				mark: {
					align: "center",
					baseline: "middle",
					fontSize: 36,
					fontWeight: 700,
					type: "text",
				},
				transform: [{ filter: "datum.kind === 'value'" }],
			},
			{
				encoding: {
					opacity: { value: 0.7 },
				},
				mark: {
					align: "center",
					baseline: "middle",
					fontSize: 12,
					type: "text",
				},
				transform: [{ filter: "datum.kind === 'label'" }],
			},
		],
		width: "container",
	}) as Spec;
