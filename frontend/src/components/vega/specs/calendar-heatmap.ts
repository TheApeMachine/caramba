import type { Spec } from "./types";

interface CalendarHeatmapOptions {
	data: Array<{ date: number | string | Date; value: number }>;
	valueTitle?: string;
	valueFormat?: string;
	colorRange?: [string, string];
	showMonthLabels?: boolean;
}

const MS_PER_DAY = 86_400_000;
const DAYS_OF_WEEK = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/*
calendarHeatmapSpec renders a GitHub-style activity grid. Columns are weeks
(anchored to the earliest sample's Sunday), rows are weekday 0–6. Caller
supplies one cell per day; missing days are filled with value=0 so the grid
stays a continuous rectangle. Month labels along the top read off the first
column where a month begins.
*/
export const calendarHeatmapSpec = ({
	data,
	valueTitle = "Value",
	valueFormat = ".0f",
	colorRange = ["var(--muted)", "var(--color-chart-1)"],
	showMonthLabels = true,
}: CalendarHeatmapOptions): Spec => {
	if (data.length === 0) {
		return {
			$schema: "https://vega.github.io/schema/vega-lite/v6.json",
			autosize: { contains: "padding", resize: true, type: "fit" },
			background: "transparent",
			data: { values: [] },
			height: "container",
			mark: "rect",
			width: "container",
		} as unknown as Spec;
	}

	const byDay = new Map<number, number>();

	for (const row of data) {
		const d = new Date(row.date);
		const key = Date.UTC(d.getFullYear(), d.getMonth(), d.getDate());
		byDay.set(key, (byDay.get(key) ?? 0) + row.value);
	}

	const sortedKeys = [...byDay.keys()].sort((a, b) => a - b);
	const firstDate = new Date(sortedKeys[0]);
	const lastDate = new Date(sortedKeys[sortedKeys.length - 1]);

	// Anchor the grid to the Sunday on or before the first sample.
	const startDay = firstDate.getUTCDay();
	const gridStart = sortedKeys[0] - startDay * MS_PER_DAY;
	const endDay = lastDate.getUTCDay();
	const gridEnd = sortedKeys[sortedKeys.length - 1] + (6 - endDay) * MS_PER_DAY;

	const cells: Array<{
		date: number;
		week: number;
		weekday: number;
		weekdayLabel: string;
		value: number;
		month: string;
		isFirstOfMonth: boolean;
	}> = [];

	for (let ts = gridStart; ts <= gridEnd; ts += MS_PER_DAY) {
		const day = new Date(ts);
		const week = Math.floor((ts - gridStart) / (7 * MS_PER_DAY));
		const weekday = day.getUTCDay();
		const value = byDay.get(ts) ?? 0;
		const dayOfMonth = day.getUTCDate();

		cells.push({
			date: ts,
			week,
			weekday,
			weekdayLabel: DAYS_OF_WEEK[weekday],
			value,
			month: day.toLocaleString(undefined, {
				month: "short",
				timeZone: "UTC",
			}),
			isFirstOfMonth: dayOfMonth <= 7 && weekday === 0,
		});
	}

	const monthMarkers = cells.filter((cell) => cell.isFirstOfMonth);

	const layers: Record<string, unknown>[] = [
		{
			encoding: {
				color: {
					field: "value",
					legend: null,
					scale: { range: colorRange },
					type: "quantitative",
				},
				tooltip: [
					{
						field: "date",
						format: "%b %d, %Y",
						title: "Date",
						type: "temporal",
					},
					{
						field: "value",
						format: valueFormat,
						title: valueTitle,
						type: "quantitative",
					},
				],
				x: {
					axis: null,
					field: "week",
					type: "ordinal",
				},
				y: {
					axis: {
						domain: false,
						labelExpr:
							"datum.value === 1 ? 'Mon' : datum.value === 3 ? 'Wed' : datum.value === 5 ? 'Fri' : ''",
						labelPadding: 4,
						ticks: false,
						title: null,
					},
					field: "weekday",
					scale: { domain: [0, 1, 2, 3, 4, 5, 6] },
					sort: null,
					type: "ordinal",
				},
			},
			mark: {
				cornerRadius: 2,
				height: { expr: "max(6, (height / 7) - 2)" },
				stroke: "var(--background)",
				strokeWidth: 1,
				type: "rect",
				width: { expr: "max(6, (width / (1 + max(1, max(datum.week)))) - 2)" },
			},
		},
	];

	if (showMonthLabels && monthMarkers.length > 0) {
		layers.push({
			data: { values: monthMarkers },
			encoding: {
				color: { value: "var(--muted-foreground)" },
				text: { field: "month", type: "nominal" },
				x: { field: "week", type: "ordinal" },
			},
			mark: {
				align: "left",
				baseline: "bottom",
				dy: -2,
				fontSize: 10,
				type: "text",
				y: -2,
			},
		});
	}

	return {
		$schema: "https://vega.github.io/schema/vega-lite/v6.json",
		autosize: { contains: "padding", resize: true, type: "fit" },
		background: "transparent",
		data: { values: cells },
		height: "container",
		layer: layers,
		padding: { bottom: 4, left: 28, right: 8, top: 16 },
		view: { stroke: null },
		width: "container",
	} as unknown as Spec;
};
