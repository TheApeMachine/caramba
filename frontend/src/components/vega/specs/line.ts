import type { Spec } from "./types";

interface LineSpecOptions {
	data: Array<Record<string, number | string>>;
	xField: string;
	seriesKeys: string[];
	seriesLabels?: Record<string, string>;
	xTitle?: string;
	yTitle?: string;
	xType?: "quantitative" | "temporal";
	zeroY?: boolean;
	yFormat?: string;
	smooth?: boolean;
	areaFill?: boolean;
	annotateLast?: boolean;
}

/*
lineSpec renders multi-series telemetry with a benchmark-friendly visual
language: soft area gradient, thick rounded stroke, a dot + value label at
the most recent sample so the current reading is always legible, and a
nearest-x hover rule with a single combined tooltip across every series.
Gridlines are dotted at low opacity so they don't fight the data.
*/
export const lineSpec = ({
	data,
	xField,
	seriesKeys,
	seriesLabels,
	xTitle,
	yTitle,
	xType = "quantitative",
	zeroY = false,
	yFormat,
	smooth = true,
	areaFill = true,
	annotateLast = true,
}: LineSpecOptions): Spec => {
	const long = data.flatMap((row) =>
		seriesKeys.map((key) => ({
			x: row[xField],
			series: seriesLabels?.[key] ?? key,
			value: typeof row[key] === "number" ? (row[key] as number) : 0,
		})),
	);

	const lastRow = data.length > 0 ? data[data.length - 1] : null;
	const lastPoints = lastRow
		? seriesKeys.map((key) => ({
				x: lastRow[xField],
				series: seriesLabels?.[key] ?? key,
				value: typeof lastRow[key] === "number" ? (lastRow[key] as number) : 0,
			}))
		: [];

	const yScale: Record<string, unknown> = { nice: true, zero: zeroY };
	const interpolate = smooth ? "monotone" : "linear";

	const colorEnc = {
		field: "series",
		legend:
			seriesKeys.length > 1
				? {
						offset: 4,
						orient: "top" as const,
						symbolType: "stroke",
						title: null,
					}
				: null,
		type: "nominal" as const,
	};

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
		scale: yScale,
		type: "quantitative" as const,
	};

	const layers: Record<string, unknown>[] = [];

	if (areaFill) {
		layers.push({
			encoding: {
				color: { field: "series", legend: null, type: "nominal" },
				opacity: { value: 0.14 },
				x: xEnc,
				y: yEnc,
			},
			mark: { interpolate, line: false, type: "area" },
		});
	}

	layers.push({
		encoding: {
			color: colorEnc,
			x: xEnc,
			y: yEnc,
		},
		mark: {
			interpolate,
			strokeCap: "round",
			strokeJoin: "round",
			strokeWidth: 2.25,
			type: "line",
		},
	});

	layers.push({
		encoding: {
			color: { value: "var(--muted-foreground)" },
			opacity: {
				condition: { empty: false, param: "hover", value: 0.55 },
				value: 0,
			},
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
		},
		mark: { strokeWidth: 1, type: "rule" },
		params: [
			{
				name: "hover",
				select: {
					clear: "pointerout",
					encodings: ["x"],
					nearest: true,
					on: "pointerover",
					type: "point",
				},
			},
		],
	});

	if (annotateLast && lastPoints.length > 0) {
		layers.push({
			data: { values: lastPoints },
			encoding: {
				color: colorEnc,
				x: xEnc,
				y: yEnc,
			},
			mark: {
				filled: true,
				opacity: 1,
				size: 90,
				stroke: "var(--background)",
				strokeWidth: 2,
				type: "point",
			},
		});

		layers.push({
			data: { values: lastPoints },
			encoding: {
				color: colorEnc,
				text: {
					field: "value",
					format: yFormat ?? ".2f",
					type: "quantitative",
				},
				x: xEnc,
				y: yEnc,
			},
			mark: {
				align: "left",
				baseline: "middle",
				dx: 8,
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
		data: { values: long },
		height: "container",
		layer: layers,
		padding: { bottom: 4, left: 4, right: 36, top: 4 },
		width: "container",
	} as unknown as Spec;
};
