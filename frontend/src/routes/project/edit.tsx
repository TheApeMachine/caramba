import { createFileRoute } from "@tanstack/react-router";

const RouteComponent = () => {
	return <div>Project edit</div>;
};

export const Route = createFileRoute("/project/edit")({
	component: RouteComponent,
});
