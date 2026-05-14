import { createFileRoute } from "@tanstack/react-router";
import { NodeGraph } from "#/components/nodegraph/component";

export const Route = createFileRoute("/nodegraph-dev")({
	component: RouteComponent,
});

function RouteComponent() {
	return <NodeGraph />;
}
