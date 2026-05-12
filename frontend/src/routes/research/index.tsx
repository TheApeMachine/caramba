import { useLiveQuery } from "@tanstack/react-db";
import { createFileRoute, Link } from "@tanstack/react-router";
import { researchProjectCollection } from "#/collections/research_project";
import { Button } from "#/components/ui/button";
import { Flex } from "#/components/ui/flex";

export const Route = createFileRoute("/research/")({
	ssr: false,
	component: ResearchIndex,
});

function ResearchIndex() {
	const { data, isLoading } = useLiveQuery((q) =>
		q.from({ project: researchProjectCollection }),
	);

	const projects = data ?? [];

	return (
		<Flex.Column gap={4} padding={6}>
			<Flex.Row className="items-center justify-between">
				<h1 className="font-semibold text-foreground text-lg">
					Research projects
				</h1>
				<Button render={<Link to="/research/new" />} size="sm">
					New project
				</Button>
			</Flex.Row>
			{isLoading ? (
				<p className="text-muted-foreground text-sm">Loading…</p>
			) : projects.length === 0 ? (
				<p className="text-muted-foreground text-sm">No projects yet.</p>
			) : (
				<ul className="flex flex-col gap-2">
					{projects.map((project) => (
						<li key={project.id}>
							<Button
								className="h-auto w-full justify-start py-3"
								render={
									<Link
										to="/research/edit"
										search={{ projectId: project.id }}
									/>
								}
								variant="outline"
							>
								<Flex.Column className="items-start gap-0.5">
									<span className="font-medium text-sm">{project.name}</span>
									{project.description ? (
										<span className="font-normal text-muted-foreground text-xs">
											{project.description}
										</span>
									) : null}
								</Flex.Column>
							</Button>
						</li>
					))}
				</ul>
			)}
		</Flex.Column>
	);
}
