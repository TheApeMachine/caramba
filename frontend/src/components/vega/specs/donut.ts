import type { Spec } from "./types";

interface DonutSpecOptions {
	data: Array<{ label: string; value: number }>;
	innerRadius?: number;
	outerRadius?: number;
	centerLabel?: string;
	centerValue?: string;
	showLegend?: boolean;
}

/*
donutSpec renders a hollow arc with optional center text. Slices have a
small gap between them (padAngle) so categories read as separated wedges
rather than a single ring with hairlines. centerLabel + centerValue draw
a stacked headline inside the hole — handy for totals.
*/
export const donutSpec = ({
	data,
	innerRadius,
	outerRadius,
	centerLabel,
	centerValue,
	showLegend = true,
}: DonutSpecOptions): Spec => {
	// Radii are expressions so the donut scales with the container. The
	// plotting `width`/`height` signals already exclude legend space, so
	// `min(width, height) / 2` is the largest radius that won't clip.
	// Caller-supplied overrides take precedence and stay fixed in pixels.
	const outer: number | { expr: string } =
		outerRadius ?? { expr: "max(20, min(width, height) / 2 - 4)" };
	const inner: number | { expr: string } =
		innerRadius ??
		(typeof outer === "number"
			? Math.round(outer * 0.64)
			: { expr: "max(0, (max(20, min(width, height) / 2 - 4)) * 0.64)" });

	const layers: Record<string, unknown>[] = [
		{
			encoding: {
				color: {
					field: "label",
					legend: showLegend
						? {
								offset: 8,
								orient: "right",
								symbolType: "circle",
								title: null,
							}
						: null,
					scale: { domain: data.map((datum) => datum.label) },
					type: "nominal",
				},
				theta: { field: "value", stack: true, type: "quantitative" },
				tooltip: [
					{ field: "label", title: "Label", type: "nominal" },
					{
						field: "value",
						format: ".0f",
						title: "Value",
						type: "quantitative",
					},
				],
			},
			mark: {
				cornerRadius: 2,
				innerRadius: inner,
				outerRadius: outer,
				padAngle: 0.015,
				stroke: "var(--background)",
				strokeWidth: 1.5,
				type: "arc",
			},
		},
	];

	if (centerValue) {
		layers.push({
			data: { values: [{ value: centerValue }] },
			encoding: {
				text: { field: "value", type: "nominal" },
			},
			mark: {
				align: "center",
				baseline: "middle",
				color: "var(--foreground)",
				fontSize: 22,
				fontWeight: 600,
				radius: 0,
				theta: 0,
				type: "text",
			},
		});
	}

	if (centerLabel) {
		layers.push({
			data: { values: [{ label: centerLabel }] },
			encoding: {
				text: { field: "label", type: "nominal" },
			},
			mark: {
				align: "center",
				baseline: "middle",
				color: "var(--muted-foreground)",
				dy: 18,
				fontSize: 10,
				radius: 0,
				theta: 0,
				type: "text",
			},
		});
	}

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		data: { values: data },
		height: "container",
		layer: layers,
		padding: { bottom: 4, left: 4, right: 4, top: 4 },
		view: { stroke: null },
		width: "container",
	} as unknown as Spec;
};
