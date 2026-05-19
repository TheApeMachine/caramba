"use client";

import { Link } from "@tanstack/react-router";
import { FolderIcon } from "lucide-react";
import { researchProjectCollection } from "#/collections/research_project";
import { Component } from "#/components/component";
import { Button } from "#/components/ui/button";

type ResearchProjectListItem = {
	id: string;
	name: string;
	description: string;
};

export const ResearchProjectsWidget = () => (
	<Component<ResearchProjectListItem[]>
		name="research projects"
		isEmpty={() => false}
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
		{(projects) => {
			if (projects.length === 0) {
				return (
					<div className="flex h-full items-center justify-center px-3 py-6 text-center text-muted-foreground text-xs">
						No projects yet. Create one from quick actions.
					</div>
				);
			}

			return (
				<ul className="flex flex-col gap-1 p-1">
					{projects.map((project) => (
						<li key={project.id}>
							<Button
								className="h-auto w-full justify-start py-2"
								render={
									<Link
										to="/research/edit"
										search={{ projectId: project.id }}
									/>
								}
								variant="ghost"
							>
								<FolderIcon className="opacity-60" />
								<div className="flex flex-col items-start gap-0.5 truncate">
									<span className="truncate font-medium text-sm">
										{project.name}
									</span>
									{project.description ? (
										<span className="truncate font-normal text-muted-foreground text-xs">
											{project.description}
										</span>
									) : null}
								</div>
							</Button>
						</li>
					))}
				</ul>
			);
		}}
	</Component>
);
