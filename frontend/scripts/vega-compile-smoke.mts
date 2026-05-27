import { compile } from "vega-lite";
import { View, parse as parseVega } from "vega";
import { lineSpec } from "../src/components/vega/specs/line.ts";
import { barSpec } from "../src/components/vega/specs/bar.ts";
import { areaSpec } from "../src/components/vega/specs/area.ts";
import { attachChartInteraction } from "../src/components/vega/interaction.ts";

const steps = [
	{ step: 0, accuracy: 0.5, loss: 1.2 },
	{ step: 1, accuracy: 0.7, loss: 0.9 },
	{ step: 2, accuracy: 0.8, loss: 0.7 },
];

const tryView = (name: string, spec: unknown) => {
	try {
		const compiled = compile(spec);
		const runtime = parseVega(compiled.spec);
		const view = new View(runtime, { renderer: "none" }).initialize();
		view.run();
		console.log(`${name}: ok`);
		view.finalize();
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		console.error(`${name}: FAIL`, message);
		if (error instanceof Error && error.stack) {
			console.error(error.stack.split("\n").slice(0, 8).join("\n"));
		}
	}
};

const line = lineSpec({
	data: steps,
	xField: "step",
	seriesKeys: ["accuracy"],
	zeroY: true,
});

const lineBase = {
	$schema: "https://vega.github.io/schema/vega-lite/v6.json",
	...(line as Record<string, unknown>),
};
delete lineBase.params;
delete lineBase.usermeta;

const area = areaSpec({
	data: steps.map((row) => ({
		date: row.step,
		desktop: row.accuracy,
		mobile: row.loss,
	})),
	seriesKeys: ["desktop", "mobile"],
});

const bar = barSpec({ data: [{ label: "A", value: 1 }] });

tryView("bar", bar);
tryView("line (with interaction)", line);
tryView("line (no interaction)", lineBase);
tryView("area", area);
