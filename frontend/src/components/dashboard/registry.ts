import {
	areaSpec,
	barSpec,
	donutSpec,
	gaugeSpec,
	metricSpec,
	type Spec,
	spiderSpec,
	stackedBarSpec,
} from "@/components/vega";

/*
WidgetDescriptor is the toolbox entry for a single widget kind. The factory
builds a fresh spec each time the widget is instantiated, so the dashboard
can mint multiple independent copies from the same toolbox tile.
*/
export interface WidgetDescriptor {
	kind: string;
	title: string;
	description: string;
	build: () => Spec;
}

const today = Date.now();
const days = (offset: number) => today - offset * 86_400_000;
const sampleSeries = Array.from({ length: 30 }, (_, i) => ({
	date: days(29 - i),
	desktop: 120 + Math.round(Math.sin(i / 3) * 40 + Math.random() * 30),
	mobile: 80 + Math.round(Math.cos(i / 4) * 30 + Math.random() * 25),
}));

const sampleBars = [
	{ label: "Mon", value: 24 },
	{ label: "Tue", value: 38 },
	{ label: "Wed", value: 31 },
	{ label: "Thu", value: 47 },
	{ label: "Fri", value: 52 },
	{ label: "Sat", value: 19 },
	{ label: "Sun", value: 12 },
];

const sampleDonut = [
	{ label: "Research", value: 42 },
	{ label: "Training", value: 28 },
	{ label: "Inference", value: 18 },
	{ label: "Idle", value: 12 },
];

const sampleSpider = [
	{ category: "Latency", value: 78 },
	{ category: "Throughput", value: 62 },
	{ category: "Accuracy", value: 91 },
	{ category: "Memory", value: 54 },
	{ category: "Cost", value: 47 },
	{ category: "Stability", value: 83 },
];

const sampleStacked = [
	{ mobile: 12, name: "Q1", server: 24, web: 18 },
	{ mobile: 18, name: "Q2", server: 31, web: 22 },
	{ mobile: 24, name: "Q3", server: 28, web: 26 },
	{ mobile: 22, name: "Q4", server: 35, web: 29 },
];

/*
widgetRegistry enumerates every widget kind the toolbox can offer. The
dashboard reads this once; adding a new widget means adding one entry.
*/
export const widgetRegistry: WidgetDescriptor[] = [
	{
		build: () =>
			areaSpec({
				data: sampleSeries,
				seriesKeys: ["desktop", "mobile"],
				seriesLabels: { desktop: "Desktop", mobile: "Mobile" },
			}),
		description: "Multi-series area chart over time",
		kind: "area",
		title: "Area",
	},
	{
		build: () => barSpec({ data: sampleBars }),
		description: "Vertical bar chart",
		kind: "bar-vertical",
		title: "Bar (vertical)",
	},
	{
		build: () => barSpec({ data: sampleBars, orientation: "horizontal" }),
		description: "Horizontal bar chart",
		kind: "bar-horizontal",
		title: "Bar (horizontal)",
	},
	{
		build: () =>
			stackedBarSpec({
				data: sampleStacked,
				seriesKeys: ["server", "web", "mobile"],
			}),
		description: "Stacked vertical bars",
		kind: "bar-stacked",
		title: "Bar (stacked)",
	},
	{
		build: () =>
			stackedBarSpec({
				data: sampleStacked,
				normalize: true,
				orientation: "horizontal",
				seriesKeys: ["server", "web", "mobile"],
			}),
		description: "Normalized stacked horizontal bars",
		kind: "bar-stacked-normalized",
		title: "Distribution",
	},
	{
		build: () => donutSpec({ data: sampleDonut }),
		description: "Donut chart with categorical breakdown",
		kind: "donut",
		title: "Donut",
	},
	{
		build: () => spiderSpec({ data: sampleSpider, seriesLabel: "Score" }),
		description: "Radar chart over multiple axes",
		kind: "spider",
		title: "Spider",
	},
	{
		build: () => gaugeSpec({ maxValue: 100, suffix: "%", value: 72 }),
		description: "Half-gauge with needle",
		kind: "gauge",
		title: "Gauge",
	},
	{
		build: () => metricSpec({ label: "Active runs", value: 1284 }),
		description: "Big-number metric card",
		kind: "metric",
		title: "Metric",
	},
];

export const widgetByKind = new Map(widgetRegistry.map((w) => [w.kind, w]));
