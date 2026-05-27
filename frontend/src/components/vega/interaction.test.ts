import { describe, expect, it } from "vitest";
import {
	buildLegendBindParam,
	buildZoomParams,
	clampDomainToBounds,
	extentNumbers,
	legendOpacityEncoding,
	wheelZoomFactor,
	zoomDomainAtPointer,
} from "#/components/vega/interaction";

describe("extentNumbers", () => {
	it("should pad a non-flat series", () => {
		expect(extentNumbers([0, 50, 100], 0)).toEqual([0, 100]);
	});

	it("should expand a flat series", () => {
		const [min, max] = extentNumbers([5, 5, 5]);
		expect(max - min).toBeGreaterThan(0);
	});
});

describe("clampDomainToBounds", () => {
	it("should not zoom out beyond the data bounds", () => {
		const bounds: [number, number] = [0, 100];
		const zoomedOut = clampDomainToBounds([-40, 140], bounds);

		expect(zoomedOut).toEqual(bounds);
	});

	it("should enforce a minimum span", () => {
		const bounds: [number, number] = [0, 100];
		const tiny = clampDomainToBounds([49.99, 50.01], bounds, 0.05);

		expect(tiny[1] - tiny[0]).toBeGreaterThanOrEqual(5);
	});
});

describe("zoomDomainAtPointer", () => {
	it("should zoom in when the wheel factor is below one", () => {
		const bounds: [number, number] = [0, 100];
		const domain: [number, number] = [0, 100];
		const zoomedIn = zoomDomainAtPointer(domain, bounds, 0.5, 0.5);

		expect(zoomedIn[1] - zoomedIn[0]).toBeLessThan(100);
	});

	it("should zoom out when the wheel factor is above one but stay within bounds", () => {
		const bounds: [number, number] = [0, 100];
		const domain: [number, number] = [25, 75];
		const zoomedOut = zoomDomainAtPointer(domain, bounds, 0.5, 2);

		expect(zoomedOut).toEqual(bounds);
	});
});

describe("buildZoomParams", () => {
	it("should use wheel zoom with shift-wheel pan and escape reset", () => {
		const params = buildZoomParams("xy");
		expect(params).toHaveLength(1);
		const zoom = params[0] as {
			select: {
				zoom: string;
				translate: string;
				clear: string;
				encodings: string[];
			};
			bind: string;
		};

		expect(zoom.bind).toBe("scales");
		expect(zoom.select.zoom).toBe("wheel!");
		expect(zoom.select.translate).toBe("wheel![event.shiftKey]");
		expect(zoom.select.clear).toBe("dblclick, escape");
		expect(zoom.select.encodings).toEqual(["x", "y"]);
	});
});

describe("buildLegendBindParam", () => {
	it("should bind a point selection to the legend", () => {
		const param = buildLegendBindParam("series") as {
			bind: string;
			select: { fields: string[] };
		};

		expect(param.bind).toBe("legend");
		expect(param.select.fields).toEqual(["series"]);
	});
});

describe("legendOpacityEncoding", () => {
	it("should dim marks when the legend selection is inactive", () => {
		expect(legendOpacityEncoding()).toEqual({
			condition: { empty: false, param: "seriesLegend", value: 1 },
			value: 0.2,
		});
	});
});

describe("wheelZoomFactor", () => {
	it("should return a factor greater than one for positive deltaY", () => {
		expect(wheelZoomFactor(120)).toBeGreaterThan(1);
	});

	it("should return a factor less than one for negative deltaY", () => {
		expect(wheelZoomFactor(-120)).toBeLessThan(1);
	});
});
