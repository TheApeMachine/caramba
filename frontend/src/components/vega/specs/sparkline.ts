import type { Spec } from "./types";

interface SparklineSpecOptions {
	values: number[];
	color?: string;
	fill?: boolean;
}

/*
sparklineSpec is a minimal trend line for inline use inside KPI cards.
No axes, no labels, no legend — just a stroke (and optional area fill)
that fits whatever container it is given.
*/
export const sparklineSpec = ({
	values,
	color = "oklch(var(--chart-1))",
	fill = true,
}: SparklineSpecOptions): Spec => {
	const data = values.map((value, index) => ({ index, value }));

	const marks: Record<string, unknown>[] = [
		{
			encoding: {
				color: { value: color },
				x: { axis: null, field: "index", type: "quantitative" },
				y: {
					axis: null,
					field: "value",
					scale: { nice: false, zero: false },
					type: "quantitative",
				},
			},
			mark: {
				interpolate: "monotone",
				strokeWidth: 1.75,
				type: "line",
			},
		},
	];

	if (fill) {
		marks.unshift({
			encoding: {
				color: { value: color },
				opacity: { value: 0.18 },
				x: { axis: null, field: "index", type: "quantitative" },
				y: {
					axis: null,
					field: "value",
					scale: { nice: false, zero: false },
					type: "quantitative",
				},
			},
			mark: { interpolate: "monotone", line: false, type: "area" },
		});
	}

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		config: {
			view: { stroke: null },
		},
		data: { values: data },
		height: "container",
		layer: marks,
		padding: 0,
		width: "container",
	} as Spec;
};
