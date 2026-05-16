import { createFileRoute } from "@tanstack/react-router";
import { IndexView } from "#/components/benchmarks";

export const Route = createFileRoute("/benchmarks/")({
	ssr: false,
	component: BenchmarksIndex,
});

function BenchmarksIndex() {
	return (
		<div className="flex h-full min-h-0 w-full flex-1 p-4">
			<IndexView />
		</div>
	);
}
