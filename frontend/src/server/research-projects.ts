import { auth } from "@clerk/tanstack-react-start/server";
import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { ResearchProject } from "#/collections/research_project";
import { backendBaseURL } from "#/lib/backend-http";

const listResearchProjectsResult = z.array(ResearchProject);

export const listResearchProjects = createServerFn({ method: "GET" }).handler(
	async () => {
		const authState = await auth();
		const token = await authState.getToken();

		if (!authState.userId || !token) {
			throw new Error("Authentication required.");
		}

		const res = await fetch(`${backendBaseURL()}/backend/research-projects`, {
			headers: { Authorization: `Bearer ${token}` },
		});

		if (!res.ok) {
			throw new Error(`Failed to fetch research projects (${res.status})`);
		}

		return listResearchProjectsResult.parse(await res.json());
	},
);

export const getResearchProject = createServerFn({ method: "GET" })
	.inputValidator((data: unknown) => z.object({ id: z.uuid() }).parse(data))
	.handler(async ({ data }) => {
		const authState = await auth();
		const token = await authState.getToken();

		if (!authState.userId || !token) {
			throw new Error("Authentication required.");
		}

		const res = await fetch(
			`${backendBaseURL()}/backend/research-projects/${data.id}`,
			{ headers: { Authorization: `Bearer ${token}` } },
		);

		if (res.status === 404) {
			return null;
		}

		if (!res.ok) {
			throw new Error(`Failed to fetch research project (${res.status})`);
		}

		return ResearchProject.parse(await res.json());
	});
