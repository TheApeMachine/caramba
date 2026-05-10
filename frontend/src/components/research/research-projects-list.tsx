import { useLiveQuery } from "@tanstack/react-db";
import { Link } from "@tanstack/react-router";
import { BookIcon, RouteIcon } from "lucide-react";
import { Button } from "#/components/ui/button";
import { Empty } from "#/components/ui/empty";
import { Flex } from "#/components/ui/flex";
import { researchProjectsCollection } from "#/lib/research-projects-collection";

/** Client-only: uses `useLiveQuery` (no SSR snapshot). Wrap with `ClientOnly` at route boundaries. */
export function ResearchProjectsList() {
	const { data, isLoading, isError } = useLiveQuery((q) =>
		q.from({ project: researchProjectsCollection }).select(({ project }) => ({
			id: project.id,
			name: project.name,
			description: project.description,
			created_at: project.created_at,
		})),
	);

	if (isLoading) {
		return (
			<Flex.Center>
				<p className="text-muted-foreground text-sm">Loading projects…</p>
			</Flex.Center>
		);
	}

	if (isError) {
		return (
			<Flex.Center className="p-4">
				<p className="max-w-md text-center text-muted-foreground text-sm">
					Could not sync research projects. Check{" "}
					<code className="text-foreground">VITE_ELECTRIC_SHAPE_URL</code> and
					your Electric shape for table{" "}
					<code className="text-foreground">research_projects</code>.
				</p>
			</Flex.Center>
		);
	}

	const projects = data ?? [];

	if (projects.length === 0) {
		return (
			<Flex.Center>
				<Empty>
					<Empty.Header>
						<Empty.Media variant="icon">
							<RouteIcon />
						</Empty.Media>
						<Empty.Title>No research projects</Empty.Title>
						<Empty.Description>
							Create a research project to get started.
						</Empty.Description>
					</Empty.Header>
					<Empty.Content>
						<Flex.Row gap={2} wrap="wrap">
							<Button render={<Link to="/research/new" />} size="sm">
								Create research project
							</Button>
							<Button
								render={<Link to="/research/paper" />}
								size="sm"
								variant="secondary"
							>
								Paper editor
							</Button>
							<Button render={<Link to="/docs" />} size="sm" variant="outline">
								<BookIcon />
								View docs
							</Button>
						</Flex.Row>
					</Empty.Content>
				</Empty>
			</Flex.Center>
		);
	}

	return (
		<Flex.Center className="items-stretch p-6">
			<div className="mx-auto flex w-full max-w-lg flex-col gap-4">
				<div className="flex flex-wrap items-center justify-between gap-2">
					<h1 className="font-semibold text-foreground text-lg">
						Research projects
					</h1>
					<Flex.Row gap={2} wrap="wrap">
						<Button
							render={<Link to="/research/paper" />}
							size="sm"
							variant="outline"
						>
							Paper editor
						</Button>
						<Button render={<Link to="/research/new" />} size="sm">
							New project
						</Button>
					</Flex.Row>
				</div>
				<ul className="flex flex-col gap-2">
					{projects.map((p) => (
						<li
							className="rounded-lg border border-border bg-card/40 px-3 py-2"
							key={p.id}
						>
							<Flex.Row className="items-start justify-between gap-2">
								<Flex.Column gap={1}>
									<p className="font-medium text-foreground text-sm">
										{p.name}
									</p>
									{p.description ? (
										<p className="text-muted-foreground text-xs">
											{p.description}
										</p>
									) : null}
								</Flex.Column>
								<Button
									render={
										<Link search={{ projectId: p.id }} to="/research/edit" />
									}
									size="sm"
									variant="outline"
								>
									Graph
								</Button>
							</Flex.Row>
						</li>
					))}
				</ul>
			</div>
		</Flex.Center>
	);
}
