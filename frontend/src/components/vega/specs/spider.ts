import type { Spec } from "./types";

interface SpiderSpecOptions {
	data: Array<{ category: string; value: number }>;
	seriesLabel?: string;
	maxValue?: number;
	rings?: number;
	ringSegments?: number;
}

/*
spiderSpec renders a radar chart. The previous full-Vega version used
runtime `signal` expressions to compute polar→Cartesian coordinates from
width/height, which proved fragile across resize and theme reflows.

This version pre-computes every coordinate in JS in a normalized space
[-1, 1] and renders through Vega-Lite's regular x/y scales. The geometry
then "just works" under resize because the scales map normalized space to
whatever pixel range the layout assigns — no signals, no dependency on
Vega's signal evaluation order.

Layers, in z-order: concentric rings → spokes → filled polygon → polygon
outline → vertex dots → axis labels.
*/
export const spiderSpec = ({
	data,
	seriesLabel = "Series",
	maxValue,
	rings = 4,
	ringSegments = 64,
}: SpiderSpecOptions): Spec => {
	const count = data.length;
	if (count === 0) {
		return {
			$schema: "https://vega.github.io/schema/vega-lite/v6.json",
			autosize: { contains: "padding", resize: true, type: "fit" },
			background: "transparent",
			data: { values: [] },
			height: "container",
			mark: "point",
			width: "container",
		} as unknown as Spec;
	}

	const max =
		maxValue ??
		Math.max(
			1,
			...data.map((datum) => (Number.isFinite(datum.value) ? datum.value : 0)),
		);

	const angleAt = (index: number) =>
		-Math.PI / 2 + (index * 2 * Math.PI) / count;

	// Normalized polygon coords (closed by appending the first vertex).
	const polygonPoints = data.map((datum, index) => {
		const angle = angleAt(index);
		const ratio = Math.max(
			0,
			Math.min(1, (Number.isFinite(datum.value) ? datum.value : 0) / max),
		);
		return {
			category: datum.category,
			nx: Math.cos(angle) * ratio,
			ny: Math.sin(angle) * ratio,
			order: index,
			ratio,
			series: seriesLabel,
			value: datum.value,
		};
	});
	polygonPoints.push({ ...polygonPoints[0], order: count });

	// One concentric ring per `rings`, traced with `ringSegments + 1` points
	// so a linear-closed line draws a smooth circle.
	const ringPoints = Array.from({ length: rings }, (_, ringIdx) => {
		const r = (ringIdx + 1) / rings;
		return Array.from({ length: ringSegments + 1 }, (_, segIdx) => {
			const t = (segIdx / ringSegments) * 2 * Math.PI - Math.PI / 2;
			return {
				nx: Math.cos(t) * r,
				ny: Math.sin(t) * r,
				order: segIdx,
				ring: ringIdx,
			};
		});
	}).flat();

	// Spoke from center to each axis tip.
	const spokes = data.flatMap((_, index) => {
		const angle = angleAt(index);
		return [
			{ axis: index, nx: 0, ny: 0, order: 0 },
			{ axis: index, nx: Math.cos(angle), ny: Math.sin(angle), order: 1 },
		];
	});

	// Labels parked just outside the outermost ring. We center-align every
	// label (one layer, simpler) and rely on the radial offset to keep them
	// off the rings. Vega-Lite text marks don't take per-datum align/baseline
	// expressions cleanly, so we trade that flexibility for one-layer simplicity.
	const labelPoints = data.map((datum, index) => {
		const angle = angleAt(index);
		return {
			category: datum.category,
			nx: Math.cos(angle) * 1.2,
			ny: Math.sin(angle) * 1.2,
		};
	});

	// Domain padded to leave room for labels outside the unit circle.
	const xEnc = {
		axis: null,
		field: "nx",
		scale: { domain: [-1.35, 1.35], nice: false, zero: false },
		type: "quantitative" as const,
	};
	const yEnc = {
		axis: null,
		field: "ny",
		scale: { domain: [-1.35, 1.35], nice: false, zero: false },
		type: "quantitative" as const,
	};

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		height: "container",
		layer: [
			// Concentric rings.
			{
				data: { values: ringPoints },
				encoding: {
					detail: { field: "ring", type: "nominal" },
					order: { field: "order", type: "quantitative" },
					x: xEnc,
					y: yEnc,
				},
				mark: {
					interpolate: "linear-closed",
					stroke: "var(--border)",
					strokeOpacity: 0.55,
					strokeWidth: 1,
					type: "line",
				},
			},
			// Spokes from center to each axis tip.
			{
				data: { values: spokes },
				encoding: {
					detail: { field: "axis", type: "nominal" },
					order: { field: "order", type: "quantitative" },
					x: xEnc,
					y: yEnc,
				},
				mark: {
					stroke: "var(--border)",
					strokeOpacity: 0.55,
					strokeWidth: 1,
					type: "line",
				},
			},
			// Filled polygon for the series. Uses a closed line mark with a
			// mark-level fill — Vega-Lite's `area` mark baselines to y=0 and
			// would render as triangular fans instead of a closed polygon.
			{
				data: { values: polygonPoints },
				encoding: {
					order: { field: "order", type: "quantitative" },
					x: xEnc,
					y: yEnc,
				},
				mark: {
					fill: "var(--color-chart-1)",
					fillOpacity: 0.22,
					interpolate: "linear-closed",
					stroke: "var(--color-chart-1)",
					strokeJoin: "round",
					strokeWidth: 2,
					type: "line",
				},
			},
			// Vertex dots.
			{
				data: { values: polygonPoints.slice(0, count) },
				encoding: {
					tooltip: [
						{ field: "category", title: "Category", type: "nominal" },
						{ field: "value", title: "Value", type: "quantitative" },
					],
					x: xEnc,
					y: yEnc,
				},
				mark: {
					fill: "var(--color-chart-1)",
					filled: true,
					size: 80,
					stroke: "var(--background)",
					strokeWidth: 1.5,
					type: "point",
				},
			},
			// Axis labels parked just outside the outermost ring.
			{
				data: { values: labelPoints },
				encoding: {
					text: { field: "category", type: "nominal" },
					x: xEnc,
					y: yEnc,
				},
				mark: {
					align: "center",
					baseline: "middle",
					fill: "var(--muted-foreground)",
					fontSize: 10,
					type: "text",
				},
			},
		],
		padding: { bottom: 4, left: 4, right: 4, top: 4 },
		view: { stroke: null },
		width: "container",
	} as unknown as Spec;
};
