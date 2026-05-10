import { createFileRoute } from "@tanstack/react-router";

const RouteComponent = () => {
	return <div>Hello "/docs/"!</div>;
};

export const Route = createFileRoute("/docs/")({
	component: RouteComponent,
});
