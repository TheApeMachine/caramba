import { createFileRoute } from "@tanstack/react-router";
import { NodeGraph } from "#/components/nodegraph/component";

export const Route = createFileRoute("/nodegraph")({
	ssr: false,
	component: NodeGraphRoute,
});

function NodeGraphRoute() {
	return (
		<div style={{ width: "100%", height: "100%", minHeight: "100vh" }}>
			<NodeGraph />
		</div>
	);
}
