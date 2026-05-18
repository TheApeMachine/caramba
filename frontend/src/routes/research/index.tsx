import { createFileRoute } from "@tanstack/react-router";
import { ResearchHub } from "#/components/research/research-hub";

function ResearchIndex() {
	return <ResearchHub />;
}

export const Route = createFileRoute("/research/")({
	ssr: false,
	component: ResearchIndex,
});
