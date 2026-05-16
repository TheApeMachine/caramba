import type { Spec } from "./types";

interface GaugeZone {
	to: number;
	color: string;
}

interface GaugeSpecOptions {
	value: number;
	maxValue?: number;
	suffix?: string;
	precision?: number;
	zones?: GaugeZone[];
	showTicks?: boolean;
	tickCount?: number;
}

/*
gaugeSpec is a half-arc dial. Visual model matches the user's reference:
each color zone has a dim background and a saturated foreground; the
foreground only paints the portion of the zone the value has reached, so
zones visibly "fill up" as value climbs. A triangular pointer sits at the
value angle on the arc rim, and the headline number is colored to match
whichever zone the value currently falls in. Tick labels ring the outside
at evenly spaced positions.

Built as a full Vega spec because the per-segment fill/empty geometry and
the pointer placement need direct mark control.
*/
export const gaugeSpec = ({
	value,
	maxValue = 100,
	suffix = "",
	precision = 0,
	zones,
	showTicks = true,
	tickCount = 11,
}: GaugeSpecOptions): Spec => {
	const max = maxValue > 0 ? maxValue : 100;
	const safeValue = Math.max(
		0,
		Math.min(max, Number.isFinite(value) ? value : 0),
	);

	const startAngle = -Math.PI / 2;
	const endAngle = Math.PI / 2;
	const span = endAngle - startAngle;

	const bands: GaugeZone[] = zones ?? [
		{ to: max * 0.5, color: "var(--color-chart-4)" },
		{ to: max * 0.7, color: "var(--color-chart-1)" },
		{ to: max * 0.85, color: "var(--muted-foreground)" },
		{ to: max, color: "var(--color-chart-3)" },
	];

	const angleAt = (v: number) =>
		startAngle + (Math.max(0, Math.min(max, v)) / max) * span;

	const valueAngle = angleAt(safeValue);

	const backgroundArcs = bands.map((band, idx) => {
		const previous = idx === 0 ? 0 : bands[idx - 1].to;
		return {
			color: band.color,
			endAngle: angleAt(band.to),
			startAngle: angleAt(previous),
		};
	});

	const foregroundArcs = bands
		.map((band, idx) => {
			const previous = idx === 0 ? 0 : bands[idx - 1].to;
			if (safeValue <= previous) return null;
			const fillTo = Math.min(safeValue, band.to);
			return {
				color: band.color,
				endAngle: angleAt(fillTo),
				startAngle: angleAt(previous),
			};
		})
		.filter(
			(arc): arc is { color: string; endAngle: number; startAngle: number } =>
				arc !== null,
		);

	const activeZone =
		bands.find((band) => safeValue <= band.to) ?? bands[bands.length - 1];

	const ticks = showTicks
		? Array.from({ length: tickCount }, (_, idx) => {
				const tickValue = (max / (tickCount - 1)) * idx;
				return {
					angle: angleAt(tickValue),
					label:
						precision > 0
							? tickValue.toFixed(precision)
							: String(Math.round(tickValue)),
				};
			})
		: [];

	const valueText =
		precision > 0
			? `${safeValue.toFixed(precision)}${suffix}`
			: `${Math.round(safeValue)}${suffix}`;

	// Radius fits whichever dimension is the binding constraint: the arc
	// is wider than tall (2:1), so width/2 caps it on landscape panels and
	// height caps it on portrait panels. centerY parks the chord-line low
	// enough that the arc occupies the available vertical space top-to-bottom.
	const radius = "max(20, min(width / 2 - 28, height - 16))";
	const centerX = "width / 2";
	const centerY = `(height + (${radius})) / 2 - 6`;
	const radiusInner = `(${radius}) * 0.66`;
	const radiusOuter = `(${radius}) * 0.94`;
	const radiusBorder = `(${radius}) * 1.0`;
	const radiusTick = `(${radius}) * 1.13`;
	const radiusPointer = `(${radius}) * 0.80`;
	const valueFontSize = `clamp(18, (${radius}) * 0.32, 48)`;

	const marks: Record<string, unknown>[] = [
		// Outer thin ring outline.
		{
			encode: {
				update: {
					endAngle: { value: endAngle },
					fill: { value: "transparent" },
					innerRadius: { signal: radiusBorder },
					outerRadius: { signal: `${radiusBorder} + 1.5` },
					startAngle: { value: startAngle },
					stroke: { value: "var(--border)" },
					strokeWidth: { value: 1 },
					x: { signal: centerX },
					y: { signal: centerY },
				},
			},
			from: { data: "track" },
			type: "arc",
		},
		// Background zones (dim).
		{
			encode: {
				update: {
					endAngle: { field: "endAngle" },
					fill: { field: "color" },
					innerRadius: { signal: radiusInner },
					opacity: { value: 0.22 },
					outerRadius: { signal: radiusOuter },
					startAngle: { field: "startAngle" },
					x: { signal: centerX },
					y: { signal: centerY },
				},
			},
			from: { data: "backgroundArcs" },
			type: "arc",
		},
		// Filled zones (saturated, only up to value).
		{
			encode: {
				update: {
					endAngle: { field: "endAngle" },
					fill: { field: "color" },
					innerRadius: { signal: radiusInner },
					outerRadius: { signal: radiusOuter },
					startAngle: { field: "startAngle" },
					x: { signal: centerX },
					y: { signal: centerY },
				},
			},
			from: { data: "foregroundArcs" },
			type: "arc",
		},
	];

	if (showTicks) {
		marks.push({
			encode: {
				update: {
					align: {
						signal:
							"abs(cos(datum.angle - PI / 2)) < 0.15 ? 'center' : (cos(datum.angle - PI / 2) > 0 ? 'left' : 'right')",
					},
					baseline: { value: "middle" },
					fill: { value: "var(--muted-foreground)" },
					fontSize: { value: 9 },
					text: { field: "label" },
					x: {
						signal: `${centerX} + cos(datum.angle - PI / 2) * ${radiusTick}`,
					},
					y: {
						signal: `${centerY} + sin(datum.angle - PI / 2) * ${radiusTick}`,
					},
				},
			},
			from: { data: "ticks" },
			type: "text",
		});
	}

	// Triangle pointer at the value angle, pointing inward toward center.
	marks.push({
		encode: {
			update: {
				angle: {
					signal: `(${valueAngle} - PI / 2) * 180 / PI + 90`,
				},
				fill: { value: "var(--background)" },
				shape: { value: "triangle" },
				size: { value: 150 },
				stroke: { value: "var(--foreground)" },
				strokeWidth: { value: 1.5 },
				x: {
					signal: `${centerX} + cos(${valueAngle} - PI / 2) * ${radiusPointer}`,
				},
				y: {
					signal: `${centerY} + sin(${valueAngle} - PI / 2) * ${radiusPointer}`,
				},
			},
		},
		type: "symbol",
	});

	// Headline value text, colored to match the active zone. Sits just
	// above the chord-line, sized proportionally to the arc radius so it
	// scales with the panel.
	marks.push({
		encode: {
			update: {
				align: { value: "center" },
				baseline: { value: "alphabetic" },
				fill: { value: activeZone.color },
				fontSize: { signal: valueFontSize },
				fontWeight: { value: 600 },
				text: { value: valueText },
				x: { signal: centerX },
				y: { signal: `${centerY} - 6` },
			},
		},
		type: "text",
	});

	return {
		$schema: "https://vega.github.io/schema/vega/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		data: [
			{ name: "track", values: [{ value: 1 }] },
			{ name: "backgroundArcs", values: backgroundArcs },
			{ name: "foregroundArcs", values: foregroundArcs },
			{ name: "ticks", values: ticks },
		],
		height: 180,
		marks,
		padding: 12,
		width: 260,
	} as unknown as Spec;
};
