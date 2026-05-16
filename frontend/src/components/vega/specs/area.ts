import type { SeriesPoint, Spec } from "./types";

interface AreaSpecOptions {
	data: SeriesPoint[];
	seriesKeys: string[];
	seriesLabels?: Record<string, string>;
	xFormat?: string;
	yFormat?: string;
	stacked?: boolean;
}

/*
areaSpec is a multi-series area chart over time. Same visual language as
lineSpec — soft fill, thick rounded stroke on top, dotted gridlines. When
stacked, areas sum at each x; otherwise they overlap with partial opacity
so trends remain readable.
*/
export const areaSpec = ({
	data,
	seriesKeys,
	seriesLabels,
	xFormat = "%b %d",
	yFormat,
	stacked = false,
}: AreaSpecOptions): Spec => {
	const vegaData = data.flatMap((point) =>
		seriesKeys.map((key) => ({
			date: point.date,
			series: seriesLabels?.[key] ?? key,
			value: point[key] ?? 0,
		})),
	);

	const xEnc = {
		axis: {
			domain: false,
			format: xFormat,
			grid: false,
			labelPadding: 6,
			tickCount: 6,
			ticks: false,
			title: null,
		},
		field: "date",
		type: "temporal" as const,
	};

	const yEnc = {
		axis: {
			domain: false,
			format: yFormat,
			grid: true,
			gridDash: [2, 4],
			gridOpacity: 0.35,
			labelPadding: 6,
			tickCount: 4,
			ticks: false,
			title: null,
		},
		field: "value",
		scale: { nice: true, zero: true },
		stack: stacked ? ("zero" as const) : null,
		type: "quantitative" as const,
	};

	const colorEnc = {
		field: "series",
		legend: {
			offset: 4,
			orient: "top" as const,
			symbolType: "circle",
			title: null,
		},
		type: "nominal" as const,
	};

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		data: { values: vegaData },
		height: "container",
		layer: [
			{
				encoding: {
					color: colorEnc,
					opacity: { value: stacked ? 0.9 : 0.45 },
					tooltip: [
						{
							field: "date",
							format: "%b %d, %Y",
							title: "Date",
							type: "temporal",
						},
						{ field: "series", title: "Series", type: "nominal" },
						{
							field: "value",
							format: yFormat ?? ".0f",
							title: "Value",
							type: "quantitative",
						},
					],
					x: xEnc,
					y: yEnc,
				},
				mark: { interpolate: "monotone", line: false, type: "area" },
			},
			{
				encoding: {
					color: colorEnc,
					x: xEnc,
					y: yEnc,
				},
				mark: {
					interpolate: "monotone",
					strokeCap: "round",
					strokeJoin: "round",
					strokeWidth: 1.75,
					type: "line",
				},
			},
		],
		padding: { bottom: 4, left: 4, right: 8, top: 4 },
		width: "container",
	} as unknown as Spec;
};
