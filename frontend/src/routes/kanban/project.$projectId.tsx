import { ClientOnly, createFileRoute } from "@tanstack/react-router";
import { KanbanBoard } from "#/components/kanban/component";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

export const Route = createFileRoute("/kanban/project/$projectId")({
	component: KanbanProjectBoardRoute,
});

function KanbanProjectBoardPending() {
	return (
		<Flex.Center className="p-6">
			<Typography.Paragraph variant="muted">
				Loading board…
			</Typography.Paragraph>
		</Flex.Center>
	);
}

function KanbanProjectBoardInner() {
	const { projectId } = Route.useParams();

	return (
		<Flex.Column className="min-h-0 flex-1 gap-4 p-4">
			<Flex.Column gap={1}>
				<h1 className="font-semibold text-foreground text-lg">
					Project Kanban
				</h1>
				<p className="break-all font-mono text-muted-foreground text-xs">
					{projectId}
				</p>
			</Flex.Column>
			<KanbanBoard scope={{ kind: "project", researchProjectId: projectId }} />
		</Flex.Column>
	);
}

function KanbanProjectBoardRoute() {
	return (
		<ClientOnly fallback={<KanbanProjectBoardPending />}>
			<KanbanProjectBoardInner />
		</ClientOnly>
	);
}
