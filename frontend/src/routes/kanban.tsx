import { createFileRoute } from "@tanstack/react-router";
import { KanbanBoard } from "#/components/kanban/component";
import { Page } from "#/components/layout/page";

const Kanban = () => {
	return (
		<>
			<Page.Header />
			<Page.Main>
				<KanbanBoard />
			</Page.Main>
		</>
	);
};

export const Route = createFileRoute("/kanban")({ component: Kanban });
