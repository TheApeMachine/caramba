import { createFileRoute, redirect } from "@tanstack/react-router";

export const Route = createFileRoute("/flume")({
	beforeLoad: () => {
		throw redirect({ to: "/nodegraph" });
	},
});
