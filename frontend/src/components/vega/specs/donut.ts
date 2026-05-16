import type { Spec } from "./types";

interface DonutSpecOptions {
	data: Array<{ label: string; value: number }>;
	innerRadius?: number;
	outerRadius?: number;
}

/*
donutSpec renders an arc chart with a hollow center. The component using it
overlays its own center-value text via plain DOM when needed; the spec only
draws the ring so it stays compositional.
*/
export const donutSpec = ({
	data,
	innerRadius = 80,
	outerRadius = 130,
}: DonutSpecOptions): Spec =>
	({
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		data: { values: data },
		background: "transparent",
		encoding: {
			color: { field: "label", legend: null, type: "nominal" },
			theta: { field: "value", type: "quantitative" },
			tooltip: [
				{ field: "label", title: "Label", type: "nominal" },
				{ field: "value", title: "Value", type: "quantitative" },
			],
		},
		height: "container",
		mark: { innerRadius, outerRadius, type: "arc" },
		width: "container",
	}) as Spec;
