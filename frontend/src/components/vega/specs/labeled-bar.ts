import type { Spec } from "./types";

interface LabeledBarOptions {
	data: Array<{ label: string; value: number }>;
	orientation?: "horizontal" | "vertical";
	valueFormat?: string;
	valueDomain?: [number, number];
	threshold?: number;
	sort?: "asc" | "desc" | null;
	axisTitle?: string;
}

/*
labeledBarSpec is a benchmark-quality bar chart: rounded bars, in-bar
value labels (rendered outside when the bar is too short to host them),
optional sort, and an optional threshold rule (e.g. 80% accuracy target).
Horizontal orientation is the default because per-class accuracy is the
primary use case and category labels read better on the y-axis.
*/
export const labeledBarSpec = ({
	data,
	orientation = "horizontal",
	valueFormat = ".0f",
	valueDomain,
	threshold,
	sort = "desc",
	axisTitle,
}: LabeledBarOptions): Spec => {
	const horizontal = orientation === "horizontal";
	const sorted = sort
		? [...data].sort((a, b) =>
				sort === "asc" ? a.value - b.value : b.value - a.value,
			)
		: data;

	const valueAxis = horizontal ? "x" : "y";
	const categoryAxis = horizontal ? "y" : "x";
	const maxValue = sorted.reduce(
		(acc, datum) => (datum.value > acc ? datum.value : acc),
		0,
	);

	const valueEnc = {
		axis: {
			domain: false,
			format: valueFormat,
			grid: horizontal,
			gridDash: [2, 4],
			gridOpacity: 0.3,
			labelPadding: 6,
			tickCount: 4,
			ticks: false,
			title: axisTitle ?? null,
		},
		field: "value",
		scale: { domain: valueDomain, nice: true, zero: true },
		type: "quantitative" as const,
	};

	const categoryEnc = {
		axis: {
			domain: false,
			labelPadding: 6,
			ticks: false,
			title: null,
		},
		field: "label",
		scale: { paddingInner: 0.25 },
		sort: sorted.map((datum) => datum.label),
		type: "nominal" as const,
	};

	const layers: Record<string, unknown>[] = [
		{
			encoding: {
				color: { value: "var(--color-chart-1)" },
				opacity: {
					condition: { empty: false, param: "barHover", value: 1 },
					value: 0.85,
				},
				tooltip: [
					{ field: "label", title: "Class", type: "nominal" },
					{
						field: "value",
						format: valueFormat,
						title: "Value",
						type: "quantitative",
					},
				],
				[valueAxis]: valueEnc,
				[categoryAxis]: categoryEnc,
			},
			mark: {
				cornerRadiusBottomRight: horizontal ? 4 : 0,
				cornerRadiusTopLeft: horizontal ? 0 : 4,
				cornerRadiusTopRight: 4,
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
		{
			encoding: {
				color: {
					condition: {
						test: `datum.value < ${maxValue * 0.35}`,
						value: "var(--foreground)",
					},
					value: "white",
				},
				text: {
					field: "value",
					format: valueFormat,
					type: "quantitative",
				},
				[valueAxis]: { field: "value", type: "quantitative" },
				[categoryAxis]: categoryEnc,
			},
			mark: {
				align: horizontal ? "right" : "center",
				baseline: horizontal ? "middle" : "bottom",
				dx: horizontal ? -6 : 0,
				dy: horizontal ? 0 : -4,
				fontSize: 10,
				fontWeight: 600,
				type: "text",
			},
		},
	];

	if (threshold !== undefined) {
		layers.push({
			data: { values: [{ threshold }] },
			encoding: {
				color: { value: "var(--color-chart-3)" },
				strokeDash: { value: [4, 3] },
				[valueAxis]: { field: "threshold", type: "quantitative" },
			},
			mark: { strokeWidth: 1, type: "rule" },
		});
	}

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		data: { values: sorted },
		height: "container",
		layer: layers,
		padding: { bottom: 4, left: 4, right: 4, top: 4 },
		width: "container",
	} as unknown as Spec;
};
