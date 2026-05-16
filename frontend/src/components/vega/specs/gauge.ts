import type { Spec } from "./types";

interface GaugeSpecOptions {
	value: number;
	maxValue?: number;
	suffix?: string;
	precision?: number;
}

/*
gaugeSpec emits a full Vega (not Vega-Lite) spec for an arc-with-needle
gauge. Vega-Lite can't express the needle geometry, so this spec uses Vega's
arc and symbol marks driven by computed signals. The widget auto-detects
Vega vs Vega-Lite from the $schema URL.
*/
export const gaugeSpec = ({
	value,
	maxValue = 100,
	suffix = "",
	precision = 0,
}: GaugeSpecOptions): Spec => {
	const normalizedMax = maxValue > 0 ? maxValue : 100;
	const normalizedValue = Number.isFinite(value) ? value : 0;

	const ticks = Array.from({ length: 6 }, (_, i) => {
		const tickValue = (normalizedMax / 5) * i;
		return { label: Math.round(tickValue).toString(), value: tickValue };
	});

	return {
		$schema: "https://vega.github.io/schema/vega/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		data: [
			{
				name: "table",
				transform: [
					{
						endAngle: { signal: "PI * 0.55" },
						field: "value",
						startAngle: { signal: "-PI * 0.55" },
						type: "pie",
					},
				],
				values: [
					{ category: 1, value: normalizedMax * 0.5 },
					{ category: 2, value: normalizedMax * 0.25 },
					{ category: 3, value: normalizedMax * 0.25 },
				],
			},
			{
				name: "needle",
				transform: [
					{
						as: "angle",
						expr: `(-PI * 0.55) + ((datum.value / ${normalizedMax}) * PI * 1.1)`,
						type: "formula",
					},
				],
				values: [{ value: normalizedValue }],
			},
			{ name: "ticks", values: ticks },
		],
		height: 220,
		marks: [
			{
				encode: {
					enter: {
						cornerRadius: { value: 3 },
						endAngle: { field: "endAngle" },
						fill: { field: "category", scale: "gaugeColor" },
						innerRadius: { signal: "min(width, height) * 0.34" },
						opacity: { value: 0.2 },
						outerRadius: { signal: "min(width, height) * 0.5" },
						padAngle: { value: 0.02 },
						startAngle: { field: "startAngle" },
						x: { signal: "width / 2" },
						y: { signal: "height - 12" },
					},
				},
				from: { data: "table" },
				type: "arc",
			},
			{
				encode: {
					enter: {
						cornerRadius: { value: 3 },
						endAngle: {
							signal: `min(datum.endAngle, (-PI * 0.55) + ((${normalizedValue} / ${normalizedMax}) * PI * 1.1))`,
						},
						fill: { field: "category", scale: "gaugeColor" },
						innerRadius: { signal: "min(width, height) * 0.34" },
						outerRadius: { signal: "min(width, height) * 0.5" },
						padAngle: { value: 0.02 },
						startAngle: { field: "startAngle" },
						x: { signal: "width / 2" },
						y: { signal: "height - 12" },
					},
				},
				from: { data: "table" },
				type: "arc",
			},
			{
				encode: {
					enter: {
						align: { value: "center" },
						baseline: { value: "top" },
						fill: { value: "oklch(var(--foreground))" },
						fontSize: { value: 28 },
						text: {
							signal:
								precision > 0
									? `format(${normalizedValue}, '.${precision}f') + '${suffix}'`
									: `format(${normalizedValue}, 'd') + '${suffix}'`,
						},
						x: { signal: "width / 2" },
						y: { signal: "height - 60" },
					},
				},
				type: "text",
			},
		],
		scales: [
			{
				domain: [1, 2, 3],
				name: "gaugeColor",
				range: [
					"oklch(var(--chart-1))",
					"oklch(var(--chart-3))",
					"oklch(var(--chart-4))",
				],
				type: "ordinal",
			},
		],
		width: 320,
	} as Spec;
};
