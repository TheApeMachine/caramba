import { ClientOnly, createFileRoute } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";
import { KanbanBoard } from "#/components/kanban/component";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

export const Route = createFileRoute("/kanban/project/$projectId")({
	component: KanbanProjectBoardRoute,
});

function KanbanProjectBoardPending() {
	const { t } = useTranslation();

	return (
		<Flex.Center className="p-6">
			<Typography.Paragraph variant="muted">
				{t("kanban.loadingBoard")}
			</Typography.Paragraph>
		</Flex.Center>
	);
}

function KanbanProjectBoardInner() {
	const { projectId } = Route.useParams();
	const { t } = useTranslation();

	return (
		<Flex.Column className="min-h-0 flex-1 gap-4 p-4">
			<Flex.Column gap={1}>
				<h1 className="font-semibold text-foreground text-lg">
					{t("kanban.projectKanban")}
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
