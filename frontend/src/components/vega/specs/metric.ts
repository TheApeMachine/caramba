import type { Spec } from "./types";

interface MetricSpecOptions {
	label: string;
	value: number | string;
	delta?: number;
	deltaSuffix?: string;
}

/*
metricSpec is a big-number card. No axes, no gridlines, no scale ticks —
just a headline value, a small caption underneath, and an optional delta
chip. Positions are signal-driven (width/2, height/2) so the layout fills
whatever container the chart widget is sized to.
*/
export const metricSpec = ({
	label,
	value,
	delta,
	deltaSuffix = "",
}: MetricSpecOptions): Spec => {
	const valueText = typeof value === "number" ? value.toLocaleString() : value;

	const marks: Record<string, unknown>[] = [
		{
			encode: {
				update: {
					align: { value: "center" },
					baseline: { value: "alphabetic" },
					fill: { value: "var(--foreground)" },
					fontSize: { value: 36 },
					fontWeight: { value: 700 },
					text: { value: valueText },
					x: { signal: "width / 2" },
					y: { signal: "height / 2 + 6" },
				},
			},
			type: "text",
		},
		{
			encode: {
				update: {
					align: { value: "center" },
					baseline: { value: "top" },
					fill: { value: "var(--muted-foreground)" },
					fontSize: { value: 11 },
					fontWeight: { value: 500 },
					text: { value: label.toUpperCase() },
					x: { signal: "width / 2" },
					y: { signal: "height / 2 + 14" },
				},
			},
			type: "text",
		},
	];

	if (delta !== undefined && Number.isFinite(delta)) {
		const positive = delta >= 0;
		const sign = positive ? "+" : "";
		marks.push({
			encode: {
				update: {
					align: { value: "center" },
					baseline: { value: "alphabetic" },
					fill: {
						value: positive ? "var(--color-chart-4)" : "var(--color-chart-3)",
					},
					fontSize: { value: 11 },
					fontWeight: { value: 600 },
					text: { value: `${sign}${delta.toFixed(2)}${deltaSuffix}` },
					x: { signal: "width / 2" },
					y: { signal: "height / 2 - 28" },
				},
			},
			type: "text",
		});
	}

	return {
		$schema: "https://vega.github.io/schema/vega/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		height: 120,
		marks,
		padding: 8,
		width: 200,
	} as unknown as Spec;
};
