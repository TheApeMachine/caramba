import type { Spec } from "./types";

interface PhasePoint {
	re: number;
	im: number;
	category?: string;
}

interface ReferenceRing {
	radius: number;
	color?: string;
	dash?: boolean;
	strokeWidth?: number;
}

interface ArcSegment {
	radius: number;
	thetaStart: number;
	thetaEnd: number;
	color: string;
	strokeWidth?: number;
}

interface VectorOverlay {
	re: number;
	im: number;
	color?: string;
	label?: string;
}

interface PhasePlotOptions {
	data: PhasePoint[];
	categoryColors?: Record<string, string>;
	categoryOrder?: string[];
	domain?: number;
	rings?: ReferenceRing[];
	arcSegments?: ArcSegment[];
	vectors?: VectorOverlay[];
	showCenterMarker?: boolean;
	xTitle?: string;
	yTitle?: string;
	valueFormat?: string;
	pointSize?: number;
}

const tracePolyline = (radius: number, segments = 96) =>
	Array.from({ length: segments + 1 }, (_, idx) => {
		const t = (idx / segments) * 2 * Math.PI;
		return { x: Math.cos(t) * radius, y: Math.sin(t) * radius };
	});

const traceArc = (
	radius: number,
	thetaStart: number,
	thetaEnd: number,
	segments = 48,
) =>
	Array.from({ length: segments + 1 }, (_, idx) => {
		const t = thetaStart + (idx / segments) * (thetaEnd - thetaStart);
		return { x: Math.cos(t) * radius, y: Math.sin(t) * radius };
	});

/*
phasePlotSpec renders a scatter on the complex Re/Im plane with several
overlays a stock scatter can't compose: concentric reference rings traced
as closed polylines, colored arc segments on a specific radius, vector
arrows from the origin, and an optional center cross marker.

Rings render through the regular x/y scales (not via `arc` marks) so they
stay perfectly aligned with data points and scale correctly under any
domain. Arrows draw as a rule from origin plus a small triangle at the
tip, oriented via expr from the rule's slope.

Symmetric `domain` (±domain) is the default so the origin lands in the
middle. If omitted, it's auto-fit to the largest |re|, |im|, ring radius,
or vector magnitude.
*/
export const phasePlotSpec = ({
	data,
	categoryColors,
	categoryOrder,
	domain,
	rings = [],
	arcSegments = [],
	vectors = [],
	showCenterMarker = true,
	xTitle = "Re",
	yTitle = "Im",
	valueFormat = ".2f",
	pointSize = 20,
}: PhasePlotOptions): Spec => {
	const maxData = data.reduce(
		(acc, point) => Math.max(acc, Math.abs(point.re), Math.abs(point.im)),
		0,
	);
	const maxRing = rings.reduce((acc, ring) => Math.max(acc, ring.radius), 0);
	const maxArc = arcSegments.reduce(
		(acc, arc) => Math.max(acc, arc.radius),
		0,
	);
	const maxVector = vectors.reduce(
		(acc, vector) => Math.max(acc, Math.hypot(vector.re, vector.im)),
		0,
	);
	const autoDomain = Math.max(1, maxData, maxRing, maxArc, maxVector) * 1.08;
	const lim = domain ?? autoDomain;

	const xEnc = {
		axis: {
			domain: false,
			format: valueFormat,
			grid: true,
			gridDash: [2, 4],
			gridOpacity: 0.3,
			labelPadding: 6,
			tickCount: 5,
			ticks: false,
			title: xTitle,
		},
		field: "x",
		scale: { domain: [-lim, lim], nice: false, zero: false },
		type: "quantitative" as const,
	};

	const yEnc = {
		axis: {
			domain: false,
			format: valueFormat,
			grid: true,
			gridDash: [2, 4],
			gridOpacity: 0.3,
			labelPadding: 6,
			tickCount: 5,
			ticks: false,
			title: yTitle,
		},
		field: "y",
		scale: { domain: [-lim, lim], nice: false, zero: false },
		type: "quantitative" as const,
	};

	const layers: Record<string, unknown>[] = [];

	// Axes through origin so the complex-plane structure reads at a glance.
	layers.push({
		data: { values: [{ x0: -lim, x1: lim, y: 0 }] },
		encoding: {
			color: { value: "var(--border)" },
			strokeOpacity: { value: 0.5 },
			x: { field: "x0", type: "quantitative" },
			x2: { field: "x1" },
			y: { field: "y", type: "quantitative" },
		},
		mark: { strokeWidth: 1, type: "rule" },
	});

	layers.push({
		data: { values: [{ x: 0, y0: -lim, y1: lim }] },
		encoding: {
			color: { value: "var(--border)" },
			strokeOpacity: { value: 0.5 },
			x: { field: "x", type: "quantitative" },
			y: { field: "y0", type: "quantitative" },
			y2: { field: "y1" },
		},
		mark: { strokeWidth: 1, type: "rule" },
	});

	rings.forEach((ring, ringIdx) => {
		const points = tracePolyline(ring.radius).map((point, order) => ({
			...point,
			order,
			ringIdx,
		}));

		layers.push({
			data: { values: points },
			encoding: {
				color: { value: ring.color ?? "var(--muted-foreground)" },
				order: { field: "order", type: "quantitative" },
				strokeDash: { value: ring.dash ? [3, 3] : [1, 0] },
				strokeOpacity: { value: 0.55 },
				x: xEnc,
				y: yEnc,
			},
			mark: {
				interpolate: "linear-closed",
				strokeWidth: ring.strokeWidth ?? 1,
				type: "line",
			},
		});
	});

	arcSegments.forEach((arc, arcIdx) => {
		const points = traceArc(arc.radius, arc.thetaStart, arc.thetaEnd).map(
			(point, order) => ({ ...point, arcIdx, order }),
		);

		layers.push({
			data: { values: points },
			encoding: {
				color: { value: arc.color },
				order: { field: "order", type: "quantitative" },
				x: xEnc,
				y: yEnc,
			},
			mark: {
				interpolate: "linear",
				strokeCap: "butt",
				strokeWidth: arc.strokeWidth ?? 6,
				type: "line",
			},
		});
	});

	const useColorEncoding =
		data.some((point) => Boolean(point.category)) || Boolean(categoryColors);

	const colorEnc = useColorEncoding
		? {
				field: "category",
				legend: {
					offset: 4,
					orient: "top-right" as const,
					symbolType: "circle",
					title: null,
				},
				scale: categoryColors
					? {
							domain: categoryOrder ?? Object.keys(categoryColors),
							range: (categoryOrder ?? Object.keys(categoryColors)).map(
								(key) => categoryColors[key] ?? "var(--color-chart-1)",
							),
						}
					: undefined,
				type: "nominal" as const,
			}
		: { value: "var(--color-chart-1)" };

	layers.push({
		data: {
			values: data.map((point) => ({
				category: point.category ?? "point",
				x: point.re,
				y: point.im,
			})),
		},
		encoding: {
			color: colorEnc,
			tooltip: [
				{
					field: "x",
					format: valueFormat,
					title: xTitle,
					type: "quantitative",
				},
				{
					field: "y",
					format: valueFormat,
					title: yTitle,
					type: "quantitative",
				},
				...(useColorEncoding
					? [{ field: "category", title: "Category", type: "nominal" as const }]
					: []),
			],
			x: xEnc,
			y: yEnc,
		},
		mark: {
			fillOpacity: 0.55,
			filled: true,
			size: pointSize,
			stroke: "var(--background)",
			strokeWidth: 0.5,
			type: "point",
		},
	});

	vectors.forEach((vector, vectorIdx) => {
		const color = vector.color ?? "var(--color-chart-2)";
		const magnitude = Math.hypot(vector.re, vector.im);
		const angle = Math.atan2(vector.im, vector.re);
		const headLen = Math.max(0.04 * lim, magnitude * 0.08);
		const baseX = vector.re - Math.cos(angle) * headLen;
		const baseY = vector.im - Math.sin(angle) * headLen;

		layers.push({
			data: { values: [{ x0: 0, x1: baseX, y0: 0, y1: baseY }] },
			encoding: {
				color: { value: color },
				x: { field: "x0", type: "quantitative" },
				x2: { field: "x1" },
				y: { field: "y0", type: "quantitative" },
				y2: { field: "y1" },
			},
			mark: { strokeWidth: 2.5, type: "rule" },
		});

		layers.push({
			data: { values: [{ x: vector.re, y: vector.im }] },
			encoding: {
				color: { value: color },
				tooltip: [
					{
						field: "x",
						format: valueFormat,
						title: xTitle,
						type: "quantitative",
					},
					{
						field: "y",
						format: valueFormat,
						title: yTitle,
						type: "quantitative",
					},
				],
				x: { field: "x", type: "quantitative", scale: xEnc.scale },
				y: { field: "y", type: "quantitative", scale: yEnc.scale },
			},
			mark: {
				angle: ((angle * 180) / Math.PI - 90 + 360) % 360,
				filled: true,
				shape: "triangle",
				size: 120,
				stroke: "var(--background)",
				strokeWidth: 0.5,
				type: "point",
			},
		});

		if (vector.label) {
			layers.push({
				data: {
					values: [{ label: vector.label, x: vector.re, y: vector.im }],
				},
				encoding: {
					color: { value: color },
					text: { field: "label", type: "nominal" },
					x: { field: "x", type: "quantitative", scale: xEnc.scale },
					y: { field: "y", type: "quantitative", scale: yEnc.scale },
				},
				mark: {
					align: "left",
					baseline: "middle",
					dx: 8,
					fontSize: 10,
					fontWeight: 600,
					type: "text",
				},
			});
		}

		void vectorIdx;
	});

	if (showCenterMarker) {
		layers.push({
			data: { values: [{ x: 0, y: 0 }] },
			encoding: {
				color: { value: "var(--foreground)" },
				x: { field: "x", type: "quantitative", scale: xEnc.scale },
				y: { field: "y", type: "quantitative", scale: yEnc.scale },
			},
			mark: {
				angle: 45,
				filled: false,
				shape: "cross",
				size: 100,
				strokeWidth: 2,
				type: "point",
			},
		});
	}

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		height: "container",
		layer: layers,
		padding: { bottom: 4, left: 4, right: 4, top: 4 },
		view: { stroke: null },
		width: "container",
	} as unknown as Spec;
};
