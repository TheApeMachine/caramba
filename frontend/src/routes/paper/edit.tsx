import { createFileRoute } from "@tanstack/react-router";
import { PaperEditorApp } from "#/components/latex/component";

const RouteComponent = () => {
	return <PaperEditorApp />;
};

export const Route = createFileRoute("/paper/edit")({
	component: RouteComponent,
});
