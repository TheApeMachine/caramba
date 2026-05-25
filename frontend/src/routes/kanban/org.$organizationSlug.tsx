import { ClientOnly, createFileRoute } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";
import { KanbanBoard } from "#/components/kanban/component";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

export const Route = createFileRoute("/kanban/org/$organizationSlug")({
	component: KanbanOrgBoardRoute,
});

function KanbanOrgBoardPending() {
	const { t } = useTranslation();

	return (
		<Flex.Center className="p-6">
			<Typography.Paragraph variant="muted">
				{t("kanban.loadingOrgBoard")}
			</Typography.Paragraph>
		</Flex.Center>
	);
}

function KanbanOrgBoardInner() {
	const { organizationSlug } = Route.useParams();
	const { t } = useTranslation();

	return (
		<Flex.Column className="min-h-0 flex-1 gap-4 p-4">
			<Flex.Column gap={1}>
				<h1 className="font-semibold text-foreground text-lg">
					{t("kanban.organizationKanban")}
				</h1>
				<p className="text-muted-foreground text-sm">{organizationSlug}</p>
			</Flex.Column>
			<KanbanBoard scope={{ kind: "aggregate", organizationSlug }} />
		</Flex.Column>
	);
}

function KanbanOrgBoardRoute() {
	return (
		<ClientOnly fallback={<KanbanOrgBoardPending />}>
			<KanbanOrgBoardInner />
		</ClientOnly>
	);
}
