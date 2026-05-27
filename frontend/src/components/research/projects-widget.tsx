"use client";

import { Link } from "@tanstack/react-router";
import { FolderIcon } from "lucide-react";
import { researchProjectCollection } from "#/collections/research_project";
import { Component } from "#/components/component";
import { Button } from "#/components/ui/button";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

type ResearchProjectListItem = {
	id: string;
	name: string;
	description: string;
};

const ProjectsEmpty = () => (
	<Flex.Center fullHeight padding={4}>
		<Typography.Paragraph variant="muted">
			No projects yet. Create one from quick actions.
		</Typography.Paragraph>
	</Flex.Center>
);

const ProjectsList = ({
	projects,
}: {
	projects: ResearchProjectListItem[];
}) => (
	<Flex.Column gap={1} padding={1} className="list-none">
		{projects.map((project) => (
			<li key={project.id}>
				<Button
					className="h-auto w-full justify-start py-2"
					render={<Link to="/research/edit" search={{ projectId: project.id }} />}
					variant="ghost"
				>
					<FolderIcon className="opacity-60" />
					<Flex.Column align="start" gap={1} className="truncate">
						<Typography.Span truncate className="font-medium text-sm">
							{project.name}
						</Typography.Span>
						{project.description ? (
							<Typography.Span
								truncate
								variant="muted"
								className="font-normal text-xs"
							>
								{project.description}
							</Typography.Span>
						) : null}
					</Flex.Column>
				</Button>
			</li>
		))}
	</Flex.Column>
);

export const ResearchProjectsWidget = () => (
	<Component<ResearchProjectListItem[]>
		name="research projects"
		empty={<ProjectsEmpty />}
		query={(query) =>
			query
				.from({ project: researchProjectCollection })
				.select(({ project }) => ({
					id: project.id,
					name: project.name,
					description: project.description,
				}))
		}
	>
		{(projects) => <ProjectsList projects={projects} />}
	</Component>
);
