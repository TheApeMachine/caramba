import type { Spec } from "./types";

interface DualAxisLineOptions {
	data: Array<Record<string, number | string>>;
	xField: string;
	leftField: string;
	rightField: string;
	leftLabel?: string;
	rightLabel?: string;
	leftColor?: string;
	rightColor?: string;
	leftFormat?: string;
	rightFormat?: string;
	xTitle?: string;
	xType?: "quantitative" | "temporal";
	rightDashed?: boolean;
}

/*
dualAxisLineSpec plots two series against the same x-axis but on
independent y-scales — useful when units don't compare (e.g. PSNR dB vs
Top-1 accuracy %, loss vs throughput). Left axis is solid, right axis is
oriented to the right and dashed by default so the eye can tell the two
series apart without a legend lookup. Both axes get their own format
spec so percentages and absolute values can coexist.
*/
export const dualAxisLineSpec = ({
	data,
	xField,
	leftField,
	rightField,
	leftLabel,
	rightLabel,
	leftColor = "var(--color-chart-1)",
	rightColor = "var(--color-chart-3)",
	leftFormat,
	rightFormat,
	xTitle,
	xType = "quantitative",
	rightDashed = true,
}: DualAxisLineOptions): Spec => {
	const xEnc = {
		axis: {
			domain: false,
			grid: false,
			labelPadding: 6,
			tickCount: 6,
			ticks: false,
			title: xTitle ?? null,
		},
		field: xField,
		scale: { nice: false },
		type: xType,
	};

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		data: { values: data },
		height: "container",
		layer: [
			{
				encoding: {
					tooltip: [
						{ field: xField, title: xTitle ?? xField, type: xType },
						{
							field: leftField,
							format: leftFormat ?? ".3f",
							title: leftLabel ?? leftField,
							type: "quantitative",
						},
					],
					x: xEnc,
					y: {
						axis: {
							domain: false,
							format: leftFormat,
							grid: true,
							gridDash: [2, 4],
							gridOpacity: 0.35,
							labelColor: leftColor,
							labelPadding: 6,
							tickCount: 4,
							ticks: false,
							title: leftLabel ?? leftField,
							titleColor: leftColor,
						},
						field: leftField,
						scale: { nice: true, zero: false },
						type: "quantitative",
					},
				},
				mark: {
					color: leftColor,
					interpolate: "monotone",
					point: { fill: leftColor, size: 30 },
					strokeCap: "round",
					strokeJoin: "round",
					strokeWidth: 2,
					type: "line",
				},
			},
			{
				encoding: {
					tooltip: [
						{ field: xField, title: xTitle ?? xField, type: xType },
						{
							field: rightField,
							format: rightFormat ?? ".3f",
							title: rightLabel ?? rightField,
							type: "quantitative",
						},
					],
					x: xEnc,
					y: {
						axis: {
							domain: false,
							format: rightFormat,
							grid: false,
							labelColor: rightColor,
							labelPadding: 6,
							orient: "right",
							tickCount: 4,
							ticks: false,
							title: rightLabel ?? rightField,
							titleColor: rightColor,
						},
						field: rightField,
						scale: { nice: true, zero: false },
						type: "quantitative",
					},
				},
				mark: {
					color: rightColor,
					interpolate: "monotone",
					point: { fill: rightColor, size: 30 },
					strokeCap: "round",
					strokeDash: rightDashed ? [5, 4] : undefined,
					strokeJoin: "round",
					strokeWidth: 2,
					type: "line",
				},
			},
		],
		padding: { bottom: 4, left: 4, right: 8, top: 4 },
		resolve: { scale: { y: "independent" } },
		width: "container",
	} as unknown as Spec;
};
