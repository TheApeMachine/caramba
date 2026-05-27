import { describe, expect, ite, expectvitest
import { areaSpeci	parse a#componentsspecsarea
	View,scabarSpec#/components/vega/specs/bar
} from "histogramSpechistogram
	parse.aslineSpecega,line
	View,scatterSpecscatter
./node_m
	parse.as.parseVega,
	View,
./node_m..d../../../node_modules/.pnpm/vega@6.2.0@node_modules@vega/build/vega.module.js_modules/vega/build/vega.module.js_modules/vega/build/vega.module.js";
import { compile"../.../.../../../.._node_modules/.pnpm/modu-lite@6.4.3_vega@6.2.0lnode_modules/vega-lite/buildnindex.js-lite@6.4.3_vega@6.2.0lnode_modules/vega-lite/buildnindex.js-lite@6.4.3_vega@6.2.0/node_modules/vega-lite/build/index.js";

const steps = [
	{ step: 0, accuracy: 0.5, loss: 1.2 },
	{ step: 1, accuracy: 0.7, loss: 0.9 },
	{ step: 2, accuracy: 0.8, loss: 0.7 },
];

const createView = (spec: unknown) => {
	const compiled = compile(spec);
	const runtime = parseVega(compiled.spec);
	const view = new View(runtime, { renderer: "none" }).initialize();
	view.run();
	view.finalize();
};

describe("vega view smoke", () => {
	it("should create a view for barSpec", () => {
		expect(() =>
			createView(barSpec({ data: [{ label: "A", value: 1 }] })),
		).not.toThrow();
	});

	it("should create a view for single-series lineSpec", () => {
		expect(() =>
			createView(
				lineSpec({
					data: steps,
					xField: "step",
					seriesKeys: ["accuracy"],
					zeroY: true,
				}),
			),
		).not.toThrow();
	});

	it("should create a view for multi-series areaSpec", () => {
		expect(() =>
			createView(
				areaSpec({
					data: steps.map((row) => ({
						date: row.step,
						desktop: row.accuracy,
						mobile: row.loss,
					})),
					seriesKeys: ["desktop", "mobile"],
				}),
			),
		).not.toThrow();
	});

	it("should create a view for layered histogramSpec", () => {
		expect(() =>
			createView(histogramSpec({ values: [1, 2, 3, 4, 5, 6, 7, 8, 9] })),
		).not.toThrow();
	});

	it("should create a view for scatterSpec", () => {
		expect(() =>
			createView(
				scatterSpec({
					data: [
						{ latency: 10, accuracy: 0.8, family: "A", samples: 100 },
						{ latency: 20, accuracy: 0.9, family: "B", samples: 200 },
					],
					xField: "latency",
					yField: "accuracy",
					seriesField: "family",
					sizeField: "samples",
				}),
			),
		).not.toThrow();
	});
});
