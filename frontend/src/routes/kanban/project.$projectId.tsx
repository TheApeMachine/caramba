import { eq, useLiveQuery } from "@tanstack/react-db";
import { ClientOnly, createFileRoute } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";
import { researchProjectCollection } from "#/collections/research_project";
import { KanbanBoard } from "#/components/kanban/component";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";
import { useBreadcrumbOverride } from "#/lib/breadcrumb-overrides";

const KanbanProjectBoardPending = () => {
	const { t } = useTranslation();

	return (
		<Flex.Center className="p-6">
			<Typography.Paragraph variant="muted">
				{t("kanban.loadingBoard")}
			</Typography.Paragraph>
		</Flex.Center>
	);
};

/*
KanbanProjectBoardHeader resolves the project name from the synced
collection so the page title shows the real project rather than a raw
UUID. Falls back to a placeholder until the live query settles.
*/
const KanbanProjectBoardHeader = ({ projectId }: { projectId: string }) => {
	const { t } = useTranslation();
	const { data } = useLiveQuery((query) =>
		query
			.from({ project: researchProjectCollection })
			.where(({ project }) => eq(project.id, projectId))
			.select(({ project }) => ({
				id: project.id,
				name: project.name,
			})),
	);

	const project = data?.[0];
	const title = project?.name ?? t("kanban.loadingBoard");

	useBreadcrumbOverride(`/kanban/project/${projectId}`, project?.name ?? null);

	return (
		<Flex.Column gap={1}>
			<Typography.Span
				className="font-medium text-muted-foreground text-xs uppercase tracking-wider"
				variant="muted"
			>
				{t("kanban.projectKanban")}
			</Typography.Span>
			<Typography.PageTitle className="text-xl tracking-tight">
				{title}
			</Typography.PageTitle>
		</Flex.Column>
	);
};

const KanbanProjectBoardInner = () => {
	const { projectId } = Route.useParams();

	return (
		<Flex.Column className="min-h-0 flex-1 gap-4 p-4">
			<KanbanProjectBoardHeader projectId={projectId} />
			<KanbanBoard scope={{ kind: "project", researchProjectId: projectId }} />
		</Flex.Column>
	);
};

const KanbanProjectBoardRoute = () => {
	return (
		<ClientOnly fallback={<KanbanProjectBoardPending />}>
			<KanbanProjectBoardInner />
		</ClientOnly>
	);
};

export const Route = createFileRoute("/kanban/project/$projectId")({
	component: KanbanProjectBoardRoute,
});
