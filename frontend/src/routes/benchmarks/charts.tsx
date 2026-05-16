"use client";

import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowLeftIcon } from "lucide-react";
import { useMemo } from "react";
import { Panel } from "#/components/benchmarks/panel";
import { Button } from "#/components/ui/button";
import { ChartWidget, VegaProvider } from "#/components/vega";
import {
	areaSpec,
	barSpec,
	boxPlotSpec,
	calendarHeatmapSpec,
	donutSpec,
	gaugeSpec,
	heatmapSpec,
	histogramSpec,
	labeledBarSpec,
	lineSpec,
	metricSpec,
	scatterSpec,
	sparklineSpec,
	spiderSpec,
	stackedBarSpec,
} from "#/components/vega/specs";

/*
Gallery is a flat catalog of every chart primitive the platform ships with
sample data. Useful when picking the right visualization for a new view, and
as a smoke test that the shared VegaProvider config still renders correctly
under whatever theme is active.
*/

const days = (offset: number) => Date.now() - offset * 86_400_000;

const seedRandom = (seed: number) => {
	let s = seed;
	return () => {
		s = (s * 9301 + 49297) % 233280;
		return s / 233280;
	};
};

const sampleSteps = () => {
	const rand = seedRandom(7);
	return Array.from({ length: 40 }, (_, idx) => ({
		step: idx + 1,
		accuracy: 0.35 + 0.55 * (1 - Math.exp(-idx / 12)) + (rand() - 0.5) * 0.04,
		loss: 1.9 * Math.exp(-idx / 14) + 0.25 + (rand() - 0.5) * 0.05,
		throughput: 780 + Math.sin(idx / 4) * 40 + rand() * 30,
	}));
};

const sampleArea = () =>
	Array.from({ length: 30 }, (_, idx) => ({
		date: days(29 - idx),
		desktop: 120 + Math.round(Math.sin(idx / 3) * 40 + Math.random() * 30),
		mobile: 80 + Math.round(Math.cos(idx / 4) * 30 + Math.random() * 25),
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

const sampleStacked = [
	{ mobile: 12, name: "Q1", server: 24, web: 18 },
	{ mobile: 18, name: "Q2", server: 31, web: 22 },
	{ mobile: 24, name: "Q3", server: 28, web: 26 },
	{ mobile: 22, name: "Q4", server: 35, web: 29 },
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

const sampleLatencies = () => {
	const rand = seedRandom(42);
	return Array.from({ length: 600 }, () => {
		// Right-skewed: most fast, long tail.
		const base = -Math.log(1 - rand()) * 0.15;
		return 1.1 + base;
	});
};

const sampleClasses = ["A", "B", "C", "D", "E"];
const sampleHeatmap = () => {
	const rand = seedRandom(11);
	return sampleClasses.flatMap((row, rowIdx) =>
		sampleClasses.map((col, colIdx) => ({
			col,
			row,
			value:
				rowIdx === colIdx
					? Math.round(180 + rand() * 30)
					: Math.round(rand() * 18),
		})),
	);
};

const samplePerClass = [
	{ label: "Positive", value: 92 },
	{ label: "Neutral", value: 71 },
	{ label: "Negative", value: 64 },
	{ label: "Sarcastic", value: 38 },
];

const sampleScatter = () => {
	const rand = seedRandom(23);
	const families = ["base", "tuned", "distilled"] as const;
	return Array.from({ length: 90 }, (_, idx) => {
		const family = families[idx % families.length];
		const familyBias = family === "tuned" ? 0.15 : family === "distilled" ? -0.05 : 0;
		const latency = 30 + rand() * 220;
		const accuracy = Math.min(
			0.99,
			Math.max(0.4, 0.55 + 0.35 * (1 - latency / 250) + familyBias + (rand() - 0.5) * 0.06),
		);
		return {
			accuracy,
			family,
			latency,
			samples: 200 + Math.round(rand() * 1800),
		};
	});
};

const sampleBoxplot = () => {
	const rand = seedRandom(31);
	const buckets = [
		{ category: "8B", base: 1.2, spread: 0.18 },
		{ category: "32B", base: 1.6, spread: 0.22 },
		{ category: "70B", base: 2.4, spread: 0.32 },
		{ category: "175B", base: 4.1, spread: 0.5 },
	];
	return buckets.flatMap((bucket) =>
		Array.from({ length: 60 }, () => {
			// Gaussian-ish via two-sample average + occasional outlier.
			const noise = (rand() + rand() - 1) * bucket.spread;
			const outlier = rand() > 0.95 ? bucket.spread * 2.5 : 0;
			return {
				category: bucket.category,
				value: bucket.base + noise + outlier,
			};
		}),
	);
};

const sampleCalendar = () => {
	const rand = seedRandom(53);
	const today = Date.now();
	return Array.from({ length: 180 }, (_, idx) => {
		const date = today - (179 - idx) * 86_400_000;
		const weekday = new Date(date).getUTCDay();
		const weekend = weekday === 0 || weekday === 6;
		const burst = rand() > 0.85 ? rand() * 40 : 0;
		return {
			date,
			value: Math.round((weekend ? 1 : 4) + rand() * 6 + burst),
		};
	});
};

const Section = ({
	title,
	children,
}: {
	title: string;
	children: React.ReactNode;
}) => (
	<section className="flex flex-col gap-3">
		<h2 className="font-semibold text-foreground text-sm uppercase tracking-wider">
			{title}
		</h2>
		<div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
			{children}
		</div>
	</section>
);

const ChartTile = ({
	title,
	hint,
	children,
	height = "h-64",
}: {
	title: string;
	hint?: string;
	children: React.ReactNode;
	height?: string;
}) => (
	<Panel title={title} hint={hint} className={height}>
		{children}
	</Panel>
);

const Gallery = () => {
	const steps = useMemo(sampleSteps, []);
	const area = useMemo(sampleArea, []);
	const latencies = useMemo(sampleLatencies, []);
	const heatmap = useMemo(sampleHeatmap, []);
	const scatter = useMemo(sampleScatter, []);
	const boxplotData = useMemo(sampleBoxplot, []);
	const calendar = useMemo(sampleCalendar, []);

	const lineMulti = useMemo(
		() =>
			lineSpec({
				data: steps as unknown as Array<Record<string, number | string>>,
				seriesKeys: ["accuracy"],
				seriesLabels: { accuracy: "Accuracy" },
				xField: "step",
				xTitle: "step",
				yFormat: ".0%",
				yTitle: "accuracy",
				zeroY: true,
			}),
		[steps],
	);

	const lineLoss = useMemo(
		() =>
			lineSpec({
				data: steps as unknown as Array<Record<string, number | string>>,
				seriesKeys: ["loss"],
				seriesLabels: { loss: "Loss" },
				xField: "step",
				xTitle: "step",
				yFormat: ".2f",
				yTitle: "loss",
				zeroY: true,
			}),
		[steps],
	);

	const areaChart = useMemo(
		() =>
			areaSpec({
				data: area,
				seriesKeys: ["desktop", "mobile"],
				seriesLabels: { desktop: "Desktop", mobile: "Mobile" },
			}),
		[area],
	);

	const barVertical = useMemo(() => barSpec({ data: sampleBars }), []);
	const barHorizontal = useMemo(
		() => barSpec({ data: sampleBars, orientation: "horizontal" }),
		[],
	);
	const stackedVertical = useMemo(
		() =>
			stackedBarSpec({
				data: sampleStacked,
				seriesKeys: ["server", "web", "mobile"],
			}),
		[],
	);
	const stackedNormalized = useMemo(
		() =>
			stackedBarSpec({
				data: sampleStacked,
				normalize: true,
				orientation: "horizontal",
				seriesKeys: ["server", "web", "mobile"],
			}),
		[],
	);

	const donut = useMemo(
		() =>
			donutSpec({
				centerLabel: "GPU hours",
				centerValue: "100",
				data: sampleDonut,
			}),
		[],
	);
	const spider = useMemo(
		() => spiderSpec({ data: sampleSpider, seriesLabel: "Score" }),
		[],
	);
	const gauge = useMemo(() => gaugeSpec({ maxValue: 100, value: 62 }), []);
	const metric = useMemo(
		() =>
			metricSpec({
				delta: 12.4,
				deltaSuffix: "%",
				label: "Active runs",
				value: 1284,
			}),
		[],
	);

	const histogram = useMemo(
		() =>
			histogramSpec({
				bins: 24,
				values: latencies,
				xFormat: ".2f",
				xTitle: "ms / sample",
			}),
		[latencies],
	);

	const heatmapChart = useMemo(
		() =>
			heatmapSpec({
				colOrder: sampleClasses,
				colTitle: "Predicted",
				data: heatmap,
				normalize: true,
				rowOrder: sampleClasses,
				rowTitle: "Actual",
			}),
		[heatmap],
	);

	const labeled = useMemo(
		() =>
			labeledBarSpec({
				axisTitle: "accuracy %",
				data: samplePerClass,
				sort: "desc",
				threshold: 80,
				valueDomain: [0, 100],
				valueFormat: ".0f",
			}),
		[],
	);

	const scatterChart = useMemo(
		() =>
			scatterSpec({
				data: scatter as unknown as Array<Record<string, number | string>>,
				seriesField: "family",
				sizeField: "samples",
				xField: "latency",
				xFormat: ".0f",
				xTitle: "latency (ms)",
				yField: "accuracy",
				yFormat: ".0%",
				yTitle: "accuracy",
			}),
		[scatter],
	);

	const boxplot = useMemo(
		() =>
			boxPlotSpec({
				categoryOrder: ["8B", "32B", "70B", "175B"],
				categoryTitle: "model size",
				data: boxplotData,
				valueFormat: ".2f",
				valueTitle: "tok/s × 1e3",
			}),
		[boxplotData],
	);

	const calendarChart = useMemo(
		() => calendarHeatmapSpec({ data: calendar, valueTitle: "runs" }),
		[calendar],
	);

	const sparkUp = useMemo(
		() => sparklineSpec({ values: steps.map((step) => step.accuracy) }),
		[steps],
	);
	const sparkDown = useMemo(
		() =>
			sparklineSpec({
				color: "var(--color-chart-3)",
				values: steps.map((step) => step.loss),
			}),
		[steps],
	);

	return (
		<VegaProvider>
			<div className="flex h-full min-h-0 w-full flex-col gap-6 overflow-y-auto p-4">
				<header className="flex flex-wrap items-start justify-between gap-3">
					<div className="flex flex-col gap-2">
						<Button
							variant="ghost"
							size="sm"
							render={<Link to="/benchmarks" />}
						>
							<ArrowLeftIcon /> Benchmarks
						</Button>
						<h1 className="font-semibold text-foreground text-xl">
							Chart gallery
						</h1>
						<p className="text-muted-foreground text-sm">
							Every primitive in <code>vega/specs/</code> rendered with sample
							data. Theme switches re-tint everything live.
						</p>
					</div>
				</header>

				<Section title="Time series">
					<ChartTile
						title="lineSpec — accuracy"
						hint="single series, 0-1 scale, last-value annotation"
					>
						<ChartWidget spec={lineMulti} />
					</ChartTile>
					<ChartTile
						title="lineSpec — loss"
						hint="auto-fit y, dotted gridlines, hover crosshair"
					>
						<ChartWidget spec={lineLoss} />
					</ChartTile>
					<ChartTile title="areaSpec" hint="multi-series area, temporal x-axis">
						<ChartWidget spec={areaChart} />
					</ChartTile>
				</Section>

				<Section title="Categorical">
					<ChartTile title="barSpec — vertical" hint="single-series bars">
						<ChartWidget spec={barVertical} />
					</ChartTile>
					<ChartTile title="barSpec — horizontal" hint="rotated orientation">
						<ChartWidget spec={barHorizontal} />
					</ChartTile>
					<ChartTile
						title="labeledBarSpec"
						hint="sorted, in-bar labels, threshold rule at 80%"
					>
						<ChartWidget spec={labeled} />
					</ChartTile>
					<ChartTile
						title="stackedBarSpec — stacked"
						hint="multiple series share a category"
					>
						<ChartWidget spec={stackedVertical} />
					</ChartTile>
					<ChartTile
						title="stackedBarSpec — normalized"
						hint="proportional, horizontal"
					>
						<ChartWidget spec={stackedNormalized} />
					</ChartTile>
				</Section>

				<Section title="Distributions & shape">
					<ChartTile
						title="histogramSpec"
						hint="binned distribution, p50/p95 reference rules"
					>
						<ChartWidget spec={histogram} />
					</ChartTile>
					<ChartTile title="heatmapSpec" hint="row-normalized confusion matrix">
						<ChartWidget spec={heatmapChart} />
					</ChartTile>
					<ChartTile title="spiderSpec" hint="multi-axis radar chart">
						<ChartWidget spec={spider} />
					</ChartTile>
					<ChartTile
						title="boxPlotSpec"
						hint="per-category whiskers, Tukey outliers"
					>
						<ChartWidget spec={boxplot} />
					</ChartTile>
				</Section>

				<Section title="Relationships">
					<ChartTile
						title="scatterSpec"
						hint="quantitative x/y, color = family, size = samples"
					>
						<ChartWidget spec={scatterChart} />
					</ChartTile>
					<ChartTile
						title="calendarHeatmapSpec"
						hint="GitHub-style daily activity grid"
					>
						<ChartWidget spec={calendarChart} />
					</ChartTile>
				</Section>

				<Section title="Parts of a whole">
					<ChartTile title="donutSpec" hint="categorical breakdown">
						<ChartWidget spec={donut} />
					</ChartTile>
					<ChartTile title="gaugeSpec" hint="full Vega spec, arc + needle">
						<ChartWidget spec={gauge} />
					</ChartTile>
				</Section>

				<Section title="Inline / KPI">
					<ChartTile
						title="metricSpec"
						hint="big-number with optional sparkline"
						height="h-44"
					>
						<ChartWidget spec={metric} />
					</ChartTile>
					<ChartTile
						title="sparklineSpec — chart-1"
						hint="strokeless trend line for KPI cards"
						height="h-44"
					>
						<ChartWidget spec={sparkUp} />
					</ChartTile>
					<ChartTile
						title="sparklineSpec — chart-3"
						hint="custom color via props"
						height="h-44"
					>
						<ChartWidget spec={sparkDown} />
					</ChartTile>
				</Section>
			</div>
		</VegaProvider>
	);
};

export const Route = createFileRoute("/benchmarks/charts")({
	component: Gallery,
	ssr: false,
});
