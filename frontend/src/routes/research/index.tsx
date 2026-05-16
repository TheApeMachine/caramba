import { createFileRoute } from "@tanstack/react-router";
import { Dashboard } from "#/components/dashboard";
import { defaultResearchLayout, researchWidgets } from "./widgets";

export const Route = createFileRoute("/research/")({
	ssr: false,
	component: ResearchIndex,
});

function ResearchIndex() {
	return (
		<div className="flex h-full min-h-0 w-full flex-1 p-4">
			<Dashboard widgets={researchWidgets} initialLayout={defaultResearchLayout} />
		</div>
	);
}
