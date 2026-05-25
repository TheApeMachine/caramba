import { useAuth } from "@clerk/tanstack-react-start";
import { useLiveQuery } from "@tanstack/react-db";
import { ClientOnly, createFileRoute, Link } from "@tanstack/react-router";
import { KanbanIcon, LayersIcon } from "lucide-react";
import { useTranslation } from "react-i18next";
import { researchProjectCollection } from "#/collections/research_project";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

const KanbanHubPending = () => {
	const { t } = useTranslation();

	return (
		<Flex.Center className="p-6">
			<Typography.Paragraph variant="muted">
				{t("kanban.loadingHub")}
			</Typography.Paragraph>
		</Flex.Center>
	);
};

function KanbanHubContent() {
	const { orgSlug } = useAuth();
	const { t } = useTranslation();

	const { data, isLoading, isError } = useLiveQuery((query) =>
		query
			.from({ project: researchProjectCollection })
			.select(({ project }) => ({
				id: project.id,
				name: project.name,
				organization_slug: project.organization_slug,
			})),
	);

	if (isLoading) {
		return <KanbanHubPending />;
	}

	if (isError) {
		return (
			<Flex.Center className="p-6">
				<Typography.Paragraph variant="muted">
					{t("kanban.loadError")}
				</Typography.Paragraph>
			</Flex.Center>
		);
	}

	const projects = data ?? [];

	return (
		<Flex.Column className="mx-auto min-h-0 w-full max-w-5xl flex-1 gap-8 p-8">
			<Flex.Column gap={2}>
				<h1 className="font-semibold text-foreground text-2xl tracking-tight">
					{t("kanban.title")}
				</h1>
				<Typography.Paragraph variant="muted">
					{t("kanban.subtitle")}
				</Typography.Paragraph>
			</Flex.Column>

			<div className="grid grid-cols-1 gap-6 md:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
				<section className="flex flex-col gap-3">
					<Flex.Row className="items-center gap-2">
						<LayersIcon
							aria-hidden
							className="size-4 shrink-0 text-muted-foreground"
						/>
						<h2 className="font-medium text-foreground text-sm">
							{t("kanban.organization")}
						</h2>
					</Flex.Row>
					<Link
						className="group flex flex-col gap-2 rounded-xl border border-border bg-card/60 p-4 transition-colors hover:border-primary/40 hover:bg-card"
						params={{ organizationSlug: orgSlug ?? "" }}
						to="/kanban/org/$organizationSlug"
					>
						<Flex.Row className="items-center gap-2">
							<KanbanIcon
								aria-hidden
								className="size-4 shrink-0 text-muted-foreground group-hover:text-primary"
							/>
							<span className="truncate font-medium text-sm">
								{orgSlug ?? t("kanban.noOrganization")}
							</span>
						</Flex.Row>
						<span className="text-muted-foreground text-xs">
							{t("kanban.aggregateDescription")}
						</span>
					</Link>
				</section>

				<section className="flex flex-col gap-3">
					<Flex.Row className="items-center gap-2">
						<KanbanIcon
							aria-hidden
							className="size-4 shrink-0 text-muted-foreground"
						/>
						<h2 className="font-medium text-foreground text-sm">
							{t("kanban.projectBoards")}
						</h2>
						<span className="ml-auto text-muted-foreground text-xs">
							{t("kanban.projectCount", { count: projects.length })}
						</span>
					</Flex.Row>
					{projects.length === 0 ? (
						<div className="rounded-xl border border-border border-dashed bg-card/30 p-6 text-center">
							<Typography.Paragraph variant="muted">
								{t("kanban.noProjects")}
							</Typography.Paragraph>
						</div>
					) : (
						<ul className="grid grid-cols-1 gap-2 sm:grid-cols-2">
							{projects.map((project) => (
								<li key={project.id}>
									<Link
										className="group flex h-full flex-col gap-1 rounded-xl border border-border bg-card/60 p-4 transition-colors hover:border-primary/40 hover:bg-card"
										params={{ projectId: project.id }}
										to="/kanban/project/$projectId"
									>
										<span className="truncate font-medium text-foreground text-sm group-hover:text-primary">
											{project.name}
										</span>
										<span className="truncate text-muted-foreground text-xs">
											{project.organization_slug ??
												t("kanban.personalUnsorted")}
										</span>
									</Link>
								</li>
							))}
						</ul>
					)}
				</section>
			</div>
		</Flex.Column>
	);
}

const KanbanIndexRoute = () => {
	return (
		<ClientOnly fallback={<KanbanHubPending />}>
			<KanbanHubContent />
		</ClientOnly>
	);
};

export const Route = createFileRoute("/kanban/")({
	component: KanbanIndexRoute,
});
