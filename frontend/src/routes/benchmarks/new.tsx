import { createFileRoute } from "@tanstack/react-router";
import { Wizard } from "#/components/benchmarks";

interface NewBenchmarkSearch {
	preset?: string;
}

export const Route = createFileRoute("/benchmarks/new")({
	ssr: false,
	component: NewBenchmarkRoute,
	validateSearch: (search): NewBenchmarkSearch => ({
		preset:
			typeof search.preset === "string" ? (search.preset as string) : undefined,
	}),
});

function NewBenchmarkRoute() {
	const { preset } = Route.useSearch();
	return (
		<div className="flex h-full min-h-0 w-full flex-1 p-4">
			<Wizard initialPresetId={preset ?? null} />
		</div>
	);
}
