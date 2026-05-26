/*
relativeTime renders a coarse "x ago" / "in x" string from an ISO date
without pulling in a date library. Resolution is per-minute below an
hour and degrades to days, months, years.
*/
const UNITS: ReadonlyArray<{ seconds: number; unit: Intl.RelativeTimeFormatUnit }> = [
	{ seconds: 60, unit: "second" },
	{ seconds: 60 * 60, unit: "minute" },
	{ seconds: 60 * 60 * 24, unit: "hour" },
	{ seconds: 60 * 60 * 24 * 30, unit: "day" },
	{ seconds: 60 * 60 * 24 * 365, unit: "month" },
	{ seconds: Number.POSITIVE_INFINITY, unit: "year" },
];

const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

export const relativeTime = (input: string | Date): string => {
	const timestamp = input instanceof Date ? input.getTime() : Date.parse(input);

	if (!Number.isFinite(timestamp)) {
		return "";
	}

	const diffSeconds = Math.round((timestamp - Date.now()) / 1000);
	const absSeconds = Math.abs(diffSeconds);

	for (let index = 0; index < UNITS.length; index++) {
		const current = UNITS[index];

		if (absSeconds < current.seconds) {
			const previous = index === 0 ? { seconds: 1 } : UNITS[index - 1];
			const value = Math.round(diffSeconds / previous.seconds);
			return formatter.format(value, current.unit);
		}
	}

	return "";
};
