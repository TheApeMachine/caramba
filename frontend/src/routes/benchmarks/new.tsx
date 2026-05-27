import { createFileRoute } from "@tanstack/react-router";
import { NewBenchmarkWizard } from "#/components/benchmarks";

interface NewBenchmarkSearch {
	preset?: string;
}

const NewBenchmarkRoute = () => {
	const { preset } = Route.useSearch();

	return (
		<div className="flex h-full min-h-0 w-full flex-1 p-4">
			<NewBenchmarkWizard initialPresetId={preset ?? null} />
		</div>
	);
};

export const Route = createFileRoute("/benchmarks/new")({
	ssr: false,
	component: NewBenchmarkRoute,
	validateSearch: (search): NewBenchmarkSearch => ({
		preset:
			typeof search.preset === "string" ? (search.preset as string) : undefined,
	}),
});
