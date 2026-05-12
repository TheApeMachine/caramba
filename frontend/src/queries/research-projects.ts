import { queryOptions } from "@tanstack/react-query";
import {
	getResearchProject,
	listResearchProjects,
} from "#/server/research-projects";

export const researchProjectsQueryOptions = queryOptions({
	queryKey: ["research-projects"],
	queryFn: () => listResearchProjects(),
});

export const researchProjectQueryOptions = (id: string) =>
	queryOptions({
		queryKey: ["research-projects", id],
		queryFn: () => getResearchProject({ data: { id } }),
	});
