import { auth } from "@clerk/tanstack-react-start/server";
import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { backendBaseURL } from "#/lib/backend-http";

const PaperBlockKindSchema = z.enum([
	"paragraph",
	"heading",
	"equation",
	"list",
]);

const HeadingPresentationSchema = z
	.enum(["abstract", "references", "acknowledgments"])
	.optional();

const PaperBlockUpsertInput = z.object({
	id: z.uuid(),
	paper_id: z.uuid(),
	sort_order: z.number().int().nonnegative(),
	kind: PaperBlockKindSchema,
	text: z.string().default(""),
	latex: z.string().default(""),
	heading_level: z.number().int().min(1).max(3).nullable().optional(),
	heading_presentation: HeadingPresentationSchema,
	list_ordered: z.boolean().default(false),
	equation_display: z.boolean().default(true),
	equation_label: z.string().default(""),
});

const PaperBlockDeleteInput = z.object({
	id: z.uuid(),
});

const PaperBlockReorderInput = z.object({
	paper_id: z.uuid(),
	entries: z
		.array(
			z.object({
				id: z.uuid(),
				sort_order: z.number().int().nonnegative(),
			}),
		)
		.min(1),
});

type BackendBody = Record<string, unknown>;

const authorizedRequest = async (
	method: "POST" | "PUT" | "DELETE",
	path: string,
	body: BackendBody,
): Promise<{ txid: number }> => {
	const authentication = await auth();
	const token = await authentication.getToken();

	if (!authentication.userId || !token) {
		throw new Error("Paper block writes require a signed-in account.");
	}

	const response = await fetch(`${backendBaseURL()}${path}`, {
		method,
		headers: {
			Authorization: `Bearer ${token}`,
			"Content-Type": "application/json",
		},
		body: JSON.stringify(body),
	});

	if (!response.ok) {
		throw new Error(
			`research-paper-blocks ${path} failed (${response.status}): ${await response.text()}`,
		);
	}

	const json = (await response.json()) as { txid?: unknown };

	if (typeof json.txid !== "number") {
		throw new Error("Paper block API must return JSON with numeric txid.");
	}

	return { txid: json.txid };
};

export const createResearchPaperBlock = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => PaperBlockUpsertInput.parse(data))
	.handler(({ data }) =>
		authorizedRequest("POST", "/backend/research-paper-blocks", data),
	);

export const updateResearchPaperBlock = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => PaperBlockUpsertInput.parse(data))
	.handler(({ data }) =>
		authorizedRequest("PUT", "/backend/research-paper-blocks", data),
	);

export const deleteResearchPaperBlock = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => PaperBlockDeleteInput.parse(data))
	.handler(({ data }) =>
		authorizedRequest("DELETE", "/backend/research-paper-blocks", data),
	);

export const reorderResearchPaperBlocks = createServerFn({ method: "POST" })
	.inputValidator((data: unknown) => PaperBlockReorderInput.parse(data))
	.handler(({ data }) =>
		authorizedRequest(
			"POST",
			"/backend/research-paper-blocks/reorder",
			data,
		),
	);

export type PaperBlockUpsertPayload = z.infer<typeof PaperBlockUpsertInput>;
export type PaperBlockReorderPayload = z.infer<typeof PaperBlockReorderInput>;
