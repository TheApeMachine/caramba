import { auth } from "@clerk/tanstack-react-start/server";
import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { ResearchProject } from "#/collections/research_project";
import { backendBaseURL } from "#/lib/backend-http";

const listResearchProjectsResult = z.array(ResearchProject);

const FETCH_TIMEOUT_MS = 10_000;

async function getAuthToken(): Promise<string> {
	const authState = await auth();
	const token = await authState.getToken();

	if (!authState.userId || !token) {
		throw new Error("Authentication required.");
	}

	return token;
}

async function authenticatedFetch(path: string, token: string): Promise<Response> {
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

	try {
		const res = await fetch(`${backendBaseURL()}${path}`, {
			headers: { Authorization: `Bearer ${token}` },
			signal: controller.signal,
		});
		clearTimeout(timer);
		return res;
	} catch (err) {
		clearTimeout(timer);
		const message = controller.signal.aborted
			? `Request to ${path} timed out after ${FETCH_TIMEOUT_MS}ms`
			: `Network error fetching ${path}: ${err instanceof Error ? err.message : String(err)}`;
		throw new Error(message, { cause: err });
	}
}

async function readErrorBody(res: Response): Promise<string> {
	try {
		const text = await res.text();
		try {
			const json = JSON.parse(text) as { message?: string };
			return json.message ?? text;
		} catch {
			return text;
		}
	} catch {
		return "";
	}
}

export const listResearchProjects = createServerFn({ method: "GET" }).handler(
	async () => {
		const token = await getAuthToken();
		const res = await authenticatedFetch("/backend/research-projects", token);

		if (!res.ok) {
			const body = await readErrorBody(res);
			throw new Error(
				`Failed to fetch research projects (${res.status})${body ? `: ${body}` : ""}`,
			);
		}

		return listResearchProjectsResult.parse(await res.json());
	},
);

export const getResearchProject = createServerFn({ method: "GET" })
	.inputValidator((data: unknown) => z.object({ id: z.uuid() }).parse(data))
	.handler(async ({ data }) => {
		const token = await getAuthToken();
		const res = await authenticatedFetch(
			`/backend/research-projects/${data.id}`,
			token,
		);

		if (res.status === 404) {
			return null;
		}

		if (!res.ok) {
			const body = await readErrorBody(res);
			throw new Error(
				`Failed to fetch research project (${res.status})${body ? `: ${body}` : ""}`,
			);
		}

		return ResearchProject.parse(await res.json());
	});
