import type { Spec } from "./types";

interface SpiderSpecOptions {
	data: Array<{ category: string; value: number }>;
	seriesLabel?: string;
}

/*
spiderSpec renders a radar/spider chart by binding value to radius and
position to theta. The closing point (loop back to start) is appended so the
polygon visually closes.
*/
export const spiderSpec = ({
	data,
	seriesLabel = "Series",
}: SpiderSpecOptions): Spec => {
	const closed = [...data, data[0]].filter(Boolean);
	const values = closed.map((point, idx) => ({
		category: point.category,
		order: idx,
		series: seriesLabel,
		value: point.value,
	}));

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		data: { values },
		background: "transparent",
		encoding: {
			color: { field: "series", legend: null, type: "nominal" },
			radius: {
				field: "value",
				scale: { type: "linear", zero: true },
				type: "quantitative",
			},
			theta: {
				field: "order",
				scale: { domain: [0, data.length] },
				type: "quantitative",
			},
			tooltip: [
				{ field: "category", title: "Category", type: "nominal" },
				{ field: "value", title: "Value", type: "quantitative" },
			],
		},
		height: "container",
		mark: { point: true, type: "line" },
		width: "container",
	} as Spec;
};
