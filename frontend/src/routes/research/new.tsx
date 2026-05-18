import { createFileRoute } from "@tanstack/react-router";
import { NewProjectWizard } from "#/components/research/new-project-wizard";

const RouteComponent = () => {
	return (
		<div className="flex h-full min-h-0 w-full flex-1 p-4">
			<NewProjectWizard />
		</div>
	);
};

export const Route = createFileRoute("/research/new")({
	ssr: false,
	component: RouteComponent,
});
