import type { Spec } from "./types";

interface HistogramSpecOptions {
	values: number[];
	bins?: number;
	xTitle?: string;
	yTitle?: string;
	xFormat?: string;
}

/*
histogramSpec bins a flat array of numbers and renders the distribution as
bars. Vega-Lite's `bin` transform handles the binning so the caller doesn't
have to. Useful for per-request latency / loss / score distributions.
*/
export const histogramSpec = ({
	values,
	bins = 30,
	xTitle,
	yTitle = "Count",
	xFormat,
}: HistogramSpecOptions): Spec =>
	({
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		data: { values: values.map((value) => ({ value })) },
		encoding: {
			tooltip: [
				{
					bin: { maxbins: bins },
					field: "value",
					format: xFormat,
					title: xTitle ?? "Bin",
					type: "quantitative",
				},
				{ aggregate: "count", title: yTitle, type: "quantitative" },
			],
			x: {
				axis: { format: xFormat, title: xTitle ?? null },
				bin: { maxbins: bins },
				field: "value",
				type: "quantitative",
			},
			y: {
				aggregate: "count",
				axis: { title: yTitle },
				type: "quantitative",
			},
		},
		height: "container",
		mark: { opacity: 0.85, type: "bar" },
		width: "container",
	}) as Spec;
