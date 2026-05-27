import type { VisualizationSpec } from "vega-embed";
import type { Spec } from "#/components/vega/specs/types";

export const CHART_INTERACTION_META_KEY = "chartInteraction";

export type ChartInteractionProfile = "none" | "x" | "y" | "xy";

export interface ChartInteractionMeta {
	profile: ChartInteractionProfile;
	bounds: {
		x?: [number, number];
		y?: [number, number];
	};
	/** When set, adds a legend-bound point selection on this nominal field. */
	legendField?: string;
}

export const SERIES_LEGEND_PARAM = "seriesLegend";

const MIN_SPAN_FRACTION = 0.02;

/*
extentNumbers returns a padded [min, max] for quantitative or temporal
values (timestamps are plain numbers). A flat series gets a synthetic span
so zoom math never divides by zero.
*/
export const extentNumbers = (
	values: number[],
	paddingRatio = 0.02,
): [number, number] => {
	const finite = values.filter(Number.isFinite);

	if (finite.length === 0) {
		return [0, 1];
	}

	let min = finite[0];
	let max = finite[0];

	for (const value of finite) {
		if (value < min) {
			min = value;
		}

		if (value > max) {
			max = value;
		}
	}

	if (min === max) {
		const pad = Math.abs(min) > 0 ? Math.abs(min) * 0.1 : 1;
		return [min - pad, max + pad];
	}

	const pad = (max - min) * paddingRatio;
	return [min - pad, max + pad];
};

/*
clampDomainToBounds keeps a zoom window inside the data extent and enforces
a minimum visible span so wheel zoom cannot collapse to a sliver.
*/
export const clampDomainToBounds = (
	domain: [number, number],
	bounds: [number, number],
	minSpanFraction = MIN_SPAN_FRACTION,
): [number, number] => {
	const [boundMin, boundMax] = bounds;
	const fullSpan = boundMax - boundMin;

	if (!Number.isFinite(fullSpan) || fullSpan <= 0) {
		return bounds;
	}

	let domainMin = domain[0];
	let domainMax = domain[1];

	if (!Number.isFinite(domainMin) || !Number.isFinite(domainMax)) {
		return bounds;
	}

	let span = domainMax - domainMin;
	const minSpan = fullSpan * minSpanFraction;

	if (span < minSpan) {
		const center = (domainMin + domainMax) / 2;
		domainMin = center - minSpan / 2;
		domainMax = center + minSpan / 2;
		span = minSpan;
	}

	if (span >= fullSpan) {
		return [boundMin, boundMax];
	}

	if (domainMin < boundMin) {
		domainMax += boundMin - domainMin;
		domainMin = boundMin;
	}

	if (domainMax > boundMax) {
		domainMin -= domainMax - boundMax;
		domainMax = boundMax;
	}

	if (domainMin < boundMin) {
		domainMin = boundMin;
		domainMax = Math.min(boundMax, domainMin + span);
	}

	if (domainMax > boundMax) {
		domainMax = boundMax;
		domainMin = Math.max(boundMin, domainMax - span);
	}

	return [domainMin, domainMax];
};

/*
zoomDomainAtPointer scales the current domain around a data-space anchor
(wheel zoom). factor > 1 zooms out; factor < 1 zooms in.
*/
export const zoomDomainAtPointer = (
	domain: [number, number],
	bounds: [number, number],
	pointerRatio: number,
	factor: number,
	minSpanFraction = MIN_SPAN_FRACTION,
): [number, number] => {
	const span = domain[1] - domain[0];
	const safeRatio = Number.isFinite(pointerRatio)
		? Math.min(1, Math.max(0, pointerRatio))
		: 0.5;
	const center = domain[0] + span * safeRatio;
	const newSpan = span * factor;
	const next: [number, number] = [center - newSpan / 2, center + newSpan / 2];

	return clampDomainToBounds(next, bounds, minSpanFraction);
};

export const wheelZoomFactor = (deltaY: number): number =>
	Math.exp(deltaY * 0.002);

/*
boundedScale pins a quantitative axis to the data extent so Vega-Lite scale
binding cannot zoom or pan into empty space beyond the series.
*/
export const boundedScale = (
	domain: [number, number],
	extras?: Record<string, unknown>,
): Record<string, unknown> => ({
	...extras,
	clamp: true,
	domain,
	nice: false,
});

/*
buildZoomParams returns Vega-Lite params for wheel zoom, shift-wheel pan,
and reset via double-click or Escape. Scale binding is handled inside the
compiled Vega view (see Vega-Lite zoom / translate / bind docs).
*/
export const buildZoomParams = (
	profile: Exclude<ChartInteractionProfile, "none">,
): Record<string, unknown>[] => {
	const encodings =
		profile === "xy" ? ["x", "y"] : profile === "y" ? ["y"] : ["x"];

	return [
		{
			name: "chartZoom",
			select: {
				clear: "dblclick, escape",
				encodings,
				mark: { fill: "transparent", stroke: "transparent" },
				translate: "wheel![event.shiftKey]",
				type: "interval",
				zoom: "wheel!",
			},
			bind: "scales",
		},
		{
			name: "chartReset",
			select: { on: "dblclick", type: "point" },
			bind: "scales",
		},
	];
};

/*
buildLegendBindParam wires legend clicks to a point selection so series
can be highlighted via encoding conditions (typically opacity).
*/
export const buildLegendBindParam = (
	field: string,
): Record<string, unknown> => ({
	name: SERIES_LEGEND_PARAM,
	select: { fields: [field], type: "point" },
	bind: "legend",
});

/*
legendOpacityEncoding dims non-selected series when the legend selection
is active; empty selection shows every series at full opacity.
*/
/** Opacity on a color encoding when legend selection is active (no extra top-level param). */
export const legendOpacityOnColorEncoding = (
	legendField: string,
	dimmed = 0.2,
) => ({
	condition: {
		selection: { type: "point", fields: [{ field: legendField }] },
		value: 1,
	},
	value: dimmed,
});

export const legendOpacityEncoding = (
	paramName = SERIES_LEGEND_PARAM,
	dimmed = 0.2,
) => ({
	condition: { empty: false, param: paramName, value: 1 },
	value: dimmed,
});

export const attachChartInteraction = (
	spec: Spec,
	meta: ChartInteractionMeta,
): Spec => {
	if (meta.profile === "none") {
		return spec;
	}

	const specRecord = spec as Record<string, unknown>;
	const existing = specRecord.usermeta;
	const existingParams = Array.isArray(specRecord.params)
		? specRecord.params
		: [];
	const interactionParams = [...buildZoomParams(meta.profile)];

	if (meta.legendField) {
		interactionParams.push(buildLegendBindParam(meta.legendField));
	}

	return {
		...spec,
		params: [...existingParams, ...interactionParams],
		usermeta: {
			...(typeof existing === "object" && existing !== null ? existing : {}),
			[CHART_INTERACTION_META_KEY]: meta,
		},
	} as Spec;
};

export const readChartInteraction = (
	spec: VisualizationSpec,
): ChartInteractionMeta | null => {
	const usermeta = (spec as Record<string, unknown>).usermeta;

	if (typeof usermeta !== "object" || usermeta === null) {
		return null;
	}

	const raw = (usermeta as Record<string, unknown>)[CHART_INTERACTION_META_KEY];

	if (typeof raw !== "object" || raw === null) {
		return null;
	}

	const record = raw as Record<string, unknown>;
	const profile = record.profile;

	if (
		profile !== "x" &&
		profile !== "y" &&
		profile !== "xy" &&
		profile !== "none"
	) {
		return null;
	}

	const boundsRaw = record.bounds;

	if (typeof boundsRaw !== "object" || boundsRaw === null) {
		return null;
	}

	const boundsRecord = boundsRaw as Record<string, unknown>;

	const readPair = (key: "x" | "y"): [number, number] | undefined => {
		const pair = boundsRecord[key];

		if (!Array.isArray(pair) || pair.length !== 2) {
			return undefined;
		}

		const left = pair[0];
		const right = pair[1];

		if (typeof left !== "number" || typeof right !== "number") {
			return undefined;
		}

		return [left, right];
	};

	const legendField =
		typeof record.legendField === "string" ? record.legendField : undefined;

	return {
		profile,
		bounds: {
			x: readPair("x"),
			y: readPair("y"),
		},
		legendField,
	};
};
