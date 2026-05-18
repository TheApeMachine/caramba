import { auth } from "@clerk/tanstack-react-start/server";
import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { backendBaseURL } from "#/lib/backend-http";

const CreatePaperInput = z.object({
	id: z.uuid(),
	research_project_id: z.uuid(),
	title: z.string().default(""),
	document: z.record(z.string(), z.unknown()).optional(),
});

const UpdatePaperInput = z.object({
	id: z.uuid(),
	expected_revision: z.coerce.number().int().positive(),
	title: z.string().default(""),
	document: z.record(z.string(), z.unknown()),
	patch: z.unknown().optional(),
	summary: z.string().default(""),
});

export type ResearchPaperDocumentState = z.infer<
	typeof UpdatePaperInput
>["document"];

async function backendPost(
	path: string,
	body: unknown,
): Promise<{ txid: number }> {
	const authentication = await auth();
	const token = await authentication.getToken();

	if (!authentication.userId || !token) {
		throw new Error("Paper writes require a signed-in account.");
	}

	const response = await fetch(`${backendBaseURL()}${path}`, {
		method: "POST",
		headers: {
			Authorization: `Bearer ${token}`,
			"Content-Type": "application/json",
		},
		body: JSON.stringify(body),
	});

	if (!response.ok) {
		throw new Error(
			`research-papers ${path} failed (${response.status}): ${await response.text()}`,
		);
	}

	const json = (await response.json()) as { txid?: unknown };

	if (typeof json.txid !== "number") {
		throw new Error("Insert API must return JSON with numeric txid.");
	}

	return { txid: json.txid };
}

async function backendPut(
	path: string,
	body: unknown,
): Promise<{ txid: number }> {
	const authentication = await auth();
	const token = await authentication.getToken();

	if (!authentication.userId || !token) {
		throw new Error("Paper writes require a signed-in account.");
	}

	const response = await fetch(`${backendBaseURL()}${path}`, {
		method: "PUT",
		headers: {
			Authorization: `Bearer ${token}`,
			"Content-Type": "application/json",
		},
		body: JSON.stringify(body),
	});

	if (response.status === 409) {
		const text = await response.text();
		throw new Error(`revision conflict (409): ${text}`);
	}

	if (!response.ok) {
		throw new Error(
			`research-papers ${path} failed (${response.status}): ${await response.text()}`,
		);
	}

	const json = (await response.json()) as { txid?: unknown };

	if (typeof json.txid !== "number") {
		throw new Error("Update API must return JSON with numeric txid.");
	}

	return { txid: json.txid };
}

export const createResearchPaper = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => CreatePaperInput.parse(data))
	.handler(async ({ data }) => {
		const payload: Record<string, unknown> = {
			id: data.id,
			research_project_id: data.research_project_id,
			title: data.title,
		};

		if (data.document !== undefined && Object.keys(data.document).length > 0) {
			payload.document = data.document;
		}

		return backendPost("/backend/research-papers", payload);
	});

export const updateResearchPaper = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => UpdatePaperInput.parse(data))
	.handler(async ({ data }) => {
		return backendPut("/backend/research-papers", {
			id: data.id,
			expected_revision: data.expected_revision,
			title: data.title,
			document: data.document,
			patch: data.patch ?? null,
			summary: data.summary,
		});
	});
