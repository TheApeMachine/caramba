import { createFileRoute } from "@tanstack/react-router";
import { Dashboard } from "#/components/dashboard";

export const Route = createFileRoute("/dashboard")({
	ssr: false,
	component: DashboardRoute,
});

function DashboardRoute() {
	return (
		<div className="flex h-full min-h-0 w-full flex-1 p-4">
			<Dashboard />
		</div>
	);
}
