import type { Spec } from "./types";

interface PhaseMarker {
	x: number | string;
	label?: string;
}

interface ShadedBand {
	x0: number | string;
	x1: number | string;
	color?: string;
	opacity?: number;
	label?: string;
}

interface Callout {
	x: number | string;
	y: number;
	text: string;
	color?: string;
}

interface AnnotatedLineOptions {
	data: Array<Record<string, number | string>>;
	xField: string;
	seriesKeys: string[];
	seriesLabels?: Record<string, string>;
	seriesColors?: Record<string, string>;
	phases?: PhaseMarker[];
	bands?: ShadedBand[];
	callouts?: Callout[];
	xTitle?: string;
	yTitle?: string;
	xType?: "quantitative" | "temporal";
	yFormat?: string;
}

/*
annotatedLineSpec renders multi-series lines with rule_shift-style
annotations: vertical phase markers at significant x positions, shaded
background bands for regimes ("SHIFT 1: Paris → Rome"), and free-form
callouts ("recover: 100%", "dip: 40%") anchored to specific (x, y)
points. Use it when the *narrative* around the curve matters as much as
the curve itself — protocol shifts, regime changes, intervention points.
*/
export const annotatedLineSpec = ({
	data,
	xField,
	seriesKeys,
	seriesLabels,
	seriesColors,
	phases = [],
	bands = [],
	callouts = [],
	xTitle,
	yTitle,
	xType = "quantitative",
	yFormat,
}: AnnotatedLineOptions): Spec => {
	const long = data.flatMap((row) =>
		seriesKeys.map((key) => ({
			color: seriesColors?.[key],
			series: seriesLabels?.[key] ?? key,
			value: typeof row[key] === "number" ? (row[key] as number) : 0,
			x: row[xField],
		})),
	);

	const xEnc = {
		axis: {
			domain: false,
			grid: false,
			labelPadding: 6,
			tickCount: 6,
			ticks: false,
			title: xTitle ?? null,
		},
		field: "x",
		scale: { nice: false },
		type: xType,
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
			title: yTitle ?? null,
		},
		field: "value",
		scale: { nice: true, zero: false },
		type: "quantitative" as const,
	};

	const colorEnc = {
		field: "series",
		legend: {
			offset: 4,
			orient: "right" as const,
			symbolType: "stroke",
			title: null,
		},
		scale: seriesColors
			? {
					domain: seriesKeys.map((key) => seriesLabels?.[key] ?? key),
					range: seriesKeys.map(
						(key) => seriesColors[key] ?? "var(--color-chart-1)",
					),
				}
			: undefined,
		type: "nominal" as const,
	};

	const layers: Record<string, unknown>[] = [];

	if (bands.length > 0) {
		const bandValues = bands.map((band, index) => ({
			color: band.color ?? "var(--color-chart-1)",
			label: band.label,
			opacity: band.opacity ?? 0.08,
			order: index,
			x0: band.x0,
			x1: band.x1,
		}));

		layers.push({
			data: { values: bandValues },
			encoding: {
				color: { field: "color", legend: null, scale: null, type: "nominal" },
				opacity: { field: "opacity", legend: null, type: "quantitative" },
				x: { field: "x0", type: xType },
				x2: { field: "x1" },
			},
			mark: { type: "rect" },
		});
	}

	layers.push({
		data: { values: long },
		encoding: {
			color: colorEnc,
			tooltip: [
				{ field: "x", title: xTitle ?? "x", type: xType },
				{ field: "series", title: "Series", type: "nominal" },
				{
					field: "value",
					format: yFormat ?? ".3f",
					title: yTitle ?? "Value",
					type: "quantitative",
				},
			],
			x: xEnc,
			y: yEnc,
		},
		mark: {
			interpolate: "monotone",
			point: { filled: true, size: 36 },
			strokeCap: "round",
			strokeJoin: "round",
			strokeWidth: 2,
			type: "line",
		},
	});

	if (phases.length > 0) {
		layers.push({
			data: { values: phases },
			encoding: {
				color: { value: "var(--muted-foreground)" },
				strokeDash: { value: [4, 4] },
				x: { field: "x", type: xType },
			},
			mark: { strokeWidth: 1, type: "rule" },
		});

		const labeledPhases = phases.filter((phase) => phase.label);

		if (labeledPhases.length > 0) {
			layers.push({
				data: { values: labeledPhases },
				encoding: {
					color: { value: "var(--muted-foreground)" },
					text: { field: "label", type: "nominal" },
					x: { field: "x", type: xType },
				},
				mark: {
					align: "center",
					baseline: "bottom",
					dy: -4,
					fontSize: 10,
					fontWeight: 600,
					type: "text",
					y: 0,
				},
			});
		}
	}

	if (callouts.length > 0) {
		layers.push({
			data: { values: callouts },
			encoding: {
				color: { field: "color", legend: null, scale: null, type: "nominal" },
				text: { field: "text", type: "nominal" },
				x: { field: "x", type: xType },
				y: { field: "y", type: "quantitative" },
			},
			mark: {
				align: "left",
				baseline: "middle",
				dx: 6,
				fontSize: 11,
				fontWeight: 600,
				type: "text",
			},
		});
	}

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		height: "container",
		layer: layers,
		padding: { bottom: 4, left: 4, right: 64, top: 16 },
		width: "container",
	} as unknown as Spec;
};
