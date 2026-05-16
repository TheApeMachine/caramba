import type { Spec } from "./types";

interface ScatterSpecOptions {
	data: Array<Record<string, number | string>>;
	xField: string;
	yField: string;
	sizeField?: string;
	seriesField?: string;
	xTitle?: string;
	yTitle?: string;
	xFormat?: string;
	yFormat?: string;
	sizeRange?: [number, number];
}

/*
scatterSpec renders a 2D point cloud over two quantitative fields. Optional
sizeField scales mark area (useful for "n samples" or "weight"), and an
optional seriesField colors points by category with a legend. Same dotted
gridlines and axis styling as lineSpec so the two read as a family. A
nearest-point hover highlights the focused mark instead of drawing a rule.
*/
export const scatterSpec = ({
	data,
	xField,
	yField,
	sizeField,
	seriesField,
	xTitle,
	yTitle,
	xFormat,
	yFormat,
	sizeRange = [30, 240],
}: ScatterSpecOptions): Spec => {
	const xEnc = {
		axis: {
			domain: false,
			format: xFormat,
			grid: true,
			gridDash: [2, 4],
			gridOpacity: 0.35,
			labelPadding: 6,
			tickCount: 6,
			ticks: false,
			title: xTitle ?? null,
		},
		field: xField,
		scale: { nice: true, zero: false },
		type: "quantitative" as const,
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
		field: yField,
		scale: { nice: true, zero: false },
		type: "quantitative" as const,
	};

	const encoding: Record<string, unknown> = {
		opacity: {
			condition: { empty: false, param: "pointHover", value: 1 },
			value: 0.7,
		},
		x: xEnc,
		y: yEnc,
	};

	if (seriesField) {
		encoding.color = {
			field: seriesField,
			legend: {
				offset: 4,
				orient: "top" as const,
				symbolType: "circle",
				title: null,
			},
			type: "nominal",
		};
	}

	if (!seriesField) {
		encoding.color = { value: "var(--color-chart-1)" };
	}

	if (sizeField) {
		encoding.size = {
			field: sizeField,
			legend: null,
			scale: { range: sizeRange },
			type: "quantitative",
		};
	}

	const tooltip: Array<Record<string, unknown>> = [
		{
			field: xField,
			format: xFormat ?? ".2f",
			title: xTitle ?? xField,
			type: "quantitative",
		},
		{
			field: yField,
			format: yFormat ?? ".2f",
			title: yTitle ?? yField,
			type: "quantitative",
		},
	];

	if (seriesField) {
		tooltip.push({ field: seriesField, title: seriesField, type: "nominal" });
	}

	if (sizeField) {
		tooltip.push({
			field: sizeField,
			format: ".2f",
			title: sizeField,
			type: "quantitative",
		});
	}

	encoding.tooltip = tooltip;

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		data: { values: data },
		encoding,
		height: "container",
		mark: {
			filled: true,
			size: sizeField ? undefined : 60,
			stroke: "var(--background)",
			strokeWidth: 1,
			type: "point",
		},
		padding: { bottom: 4, left: 4, right: 8, top: 4 },
		params: [
			{
				name: "pointHover",
				select: {
					clear: "pointerout",
					nearest: true,
					on: "pointerover",
					type: "point",
				},
			},
		],
		width: "container",
	} as unknown as Spec;
};
