import { useLiveQuery } from "@tanstack/react-db";
import { ClientOnly, createFileRoute, Link } from "@tanstack/react-router";
import { KanbanIcon, LayersIcon } from "lucide-react";
import { Button } from "#/components/ui/button";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";
import { researchProjectsCollection } from "#/lib/research-projects-collection";

const KanbanHubPending = () => {
	return (
		<Flex.Center className="p-6">
			<Typography.Paragraph variant="muted">
				Loading Kanban hub…
			</Typography.Paragraph>
		</Flex.Center>
	);
};

function KanbanHubContent() {
	const defaultOrganizationSlug =
		import.meta.env.VITE_CLERK_ORGANIZATION_SLUG ?? "caramba";

	const { data, isLoading, isError } = useLiveQuery((query) =>
		query
			.from({ project: researchProjectsCollection })
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
					Could not load projects for Kanban routing.
				</Typography.Paragraph>
			</Flex.Center>
		);
	}

	const projects = data ?? [];

	return (
		<Flex.Column className="mx-auto min-h-0 w-full max-w-lg flex-1 gap-6 p-6">
			<Flex.Column gap={2}>
				<h1 className="font-semibold text-foreground text-lg">Kanban</h1>
				<Typography.Paragraph variant="muted">
					Open a board per research project, or the aggregate board for every
					project tagged with an organization slug.
				</Typography.Paragraph>
			</Flex.Column>

			<Flex.Column className="gap-3 rounded-xl border border-border bg-card/40 p-4">
				<Flex.Row className="items-center gap-2">
					<LayersIcon
						aria-hidden
						className="size-4 shrink-0 text-muted-foreground"
					/>
					<h2 className="font-medium text-foreground text-sm">
						Organization boards
					</h2>
				</Flex.Row>
				<Button
					className="justify-start gap-3"
					render={
						<Link
							params={{ organizationSlug: defaultOrganizationSlug }}
							to="/kanban/org/$organizationSlug"
						/>
					}
					variant="outline"
				>
					<KanbanIcon aria-hidden className="size-4 shrink-0" />
					<Flex.Column className="items-start gap-0.5">
						<span className="font-medium text-sm">
							{defaultOrganizationSlug}
						</span>
						<span className="font-normal text-muted-foreground text-xs">
							All projects with this organization slug
						</span>
					</Flex.Column>
				</Button>
			</Flex.Column>

			<Flex.Column className="gap-3">
				<h2 className="font-medium text-foreground text-sm">Project boards</h2>
				{projects.length === 0 ? (
					<Typography.Paragraph variant="muted">
						No research projects synced yet.
					</Typography.Paragraph>
				) : (
					<ul className="flex flex-col gap-2">
						{projects.map((project) => (
							<li key={project.id}>
								<Button
									className="h-auto w-full justify-between gap-3 py-3"
									render={
										<Link
											params={{ projectId: project.id }}
											to="/kanban/project/$projectId"
										/>
									}
									variant="outline"
								>
									<Flex.Column className="min-w-0 items-start gap-0.5">
										<span className="truncate font-medium text-sm">
											{project.name}
										</span>
										<span className="font-normal text-muted-foreground text-xs">
											{project.organization_slug
												? `Org slug: ${project.organization_slug}`
												: "Personal / unsorted"}
										</span>
									</Flex.Column>
									<KanbanIcon aria-hidden className="size-4 shrink-0" />
								</Button>
							</li>
						))}
					</ul>
				)}
			</Flex.Column>
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
