import { chat, toServerSentEventsResponse, toolDefinition } from "@tanstack/ai";
import { OPENAI_CHAT_MODELS, openaiText } from "@tanstack/ai-openai";
import { createFileRoute } from "@tanstack/react-router";

const RATE_LIMIT_WINDOW_MS = 60_000;
const RATE_LIMIT_MAX = 120;
const streamTimeoutMsRaw = Number.parseInt(
	process.env.ASSISTANT_STREAM_TIMEOUT_MS ?? "",
	10,
);
const STREAM_TIMEOUT_MS =
	Number.isFinite(streamTimeoutMsRaw) && streamTimeoutMsRaw > 0
		? streamTimeoutMsRaw
		: 120_000;

const buckets = new Map<string, number[]>();

function pruneAndCount(timestamps: number[], nowMs: number) {
	let index = 0;
	while (index < timestamps.length && nowMs - timestamps[index] > RATE_LIMIT_WINDOW_MS) {
		index += 1;
	}
	timestamps.splice(0, index);
	timestamps.push(nowMs);
	return timestamps.length;
}

function getClientFingerprint(request: Request) {
	const forwarded = request.headers.get("x-forwarded-for");
	if (forwarded) return forwarded.split(",")[0]?.trim() ?? "unknown-forwarded";
	return request.headers.get("fly-client-ip") ?? "unknown-peer";
}

function logAssistantEvent(details: Record<string, unknown>) {
	console.error(JSON.stringify({ component: "api/assistant", ...details }));
}

function isUIMessage(candidate: unknown) {
	return (
		typeof candidate === "object" &&
		candidate !== null &&
		typeof (candidate as { id?: unknown }).id === "string" &&
		typeof (candidate as { role?: unknown }).role === "string" &&
		Array.isArray((candidate as { parts?: unknown }).parts)
	);
}

type SupportedOpenAIModel = (typeof OPENAI_CHAT_MODELS)[number];

function coerceOpenAIModel(
	requestedRaw: string | undefined,
	opts: { requester: string },
): SupportedOpenAIModel {
	const fallback: SupportedOpenAIModel = "gpt-4o-mini";
	const trimmed = requestedRaw?.trim();
	if (!trimmed) return fallback;
	if (!(OPENAI_CHAT_MODELS as ReadonlyArray<string>).includes(trimmed)) {
		logAssistantEvent({
			level: "warn",
			message: `Unknown OPENAI_MODEL "${trimmed}", falling back to ${fallback}`,
			requester: opts.requester,
		});
		return fallback;
	}
	return trimmed as SupportedOpenAIModel;
}

/*
PAPER_EDITOR_TOOL_SCHEMAS defines the server-side stubs for all client-executed
paper editor tools. The model sees their schemas and can call them; execution
happens on the client via the tool-call stream events.
*/
const PAPER_EDITOR_TOOL_SCHEMAS = {
	paper_list_blocks: toolDefinition({
		name: "paper_list_blocks",
		description: "Returns the current list of blocks in the paper with their IDs, types, and content. Call this first before inserting or editing blocks.",
		inputSchema: { type: "object" as const, properties: {} },
	}),
	paper_update_metadata: toolDefinition({
		name: "paper_update_metadata",
		description: "Updates the paper metadata (title, authors, abstract, keywords). Pass only the fields you want to change.",
		inputSchema: {
			type: "object" as const,
			properties: {
				title:    { type: "string" },
				authors:  { type: "string", description: "One author per line." },
				abstract: { type: "string" },
				keywords: { type: "string", description: "Comma-separated." },
			},
		},
	}),
	paper_insert_block: toolDefinition({
		name: "paper_insert_block",
		description: "Inserts a new block (paragraph, heading, or equation) after the given block ID. Use afterId 'last' to append at end.",
		inputSchema: {
			type: "object" as const,
			required: ["afterId", "blockType"],
			properties: {
				afterId:   { type: "string", description: "ID of the block to insert after, or 'last'." },
				blockType: { type: "string", enum: ["paragraph", "heading", "equation"] },
				text:      { type: "string" },
				level:     { type: "number", enum: [1, 2, 3] },
				latex:     { type: "string" },
			},
		},
	}),
	paper_update_block: toolDefinition({
		name: "paper_update_block",
		description: "Updates the content of an existing block by its ID.",
		inputSchema: {
			type: "object" as const,
			required: ["id"],
			properties: {
				id:    { type: "string" },
				text:  { type: "string" },
				latex: { type: "string" },
			},
		},
	}),
	paper_remove_block: toolDefinition({
		name: "paper_remove_block",
		description: "Removes a block from the paper by its ID.",
		inputSchema: {
			type: "object" as const,
			required: ["id"],
			properties: { id: { type: "string" } },
		},
	}),
	paper_scroll_to_block: toolDefinition({
		name: "paper_scroll_to_block",
		description: "Scrolls the paper editor to bring a specific block into view. Use this to direct the user's attention.",
		inputSchema: {
			type: "object" as const,
			required: ["id"],
			properties: { id: { type: "string" } },
		},
	}),
} as const;

type ToolName = keyof typeof PAPER_EDITOR_TOOL_SCHEMAS;

function resolveTools(requested: unknown) {
	if (!Array.isArray(requested) || requested.length === 0) return undefined;
	const valid = requested.filter(
		(n): n is ToolName => typeof n === "string" && n in PAPER_EDITOR_TOOL_SCHEMAS,
	);
	return valid.length > 0 ? valid.map((n) => PAPER_EDITOR_TOOL_SCHEMAS[n]) : undefined;
}

export const Route = createFileRoute("/api/assistant")({
	server: {
		handlers: {
			POST: async ({ request }) => {
				const requesterFingerprint = getClientFingerprint(request);

				try {
					const bearerSecret = process.env.ASSISTANT_BEARER_SECRET;
					if (bearerSecret !== undefined && bearerSecret.length > 0) {
						const authHeader = request.headers.get("authorization") ?? "";
						if (authHeader !== `Bearer ${bearerSecret}`) {
							return new Response(JSON.stringify({ error: "Unauthorized" }), {
								status: 401,
								headers: { "Content-Type": "application/json" },
							});
						}
					}

					const nowBucket = Date.now();
					const history = buckets.get(requesterFingerprint) ?? [];
					const rateCount = pruneAndCount(history, nowBucket);
					buckets.set(requesterFingerprint, history);

					if (rateCount > RATE_LIMIT_MAX) {
						return new Response(JSON.stringify({ error: "Too many requests" }), {
							status: 429,
							headers: { "Content-Type": "application/json" },
						});
					}

					if (!process.env.OPENAI_API_KEY) {
						return new Response(
							JSON.stringify({ error: "OPENAI_API_KEY not configured" }),
							{ status: 503, headers: { "Content-Type": "application/json" } },
						);
					}

					let payload: Record<string, unknown>;
					try {
						payload = (await request.json()) as Record<string, unknown>;
					} catch {
						return new Response(JSON.stringify({ error: "Invalid JSON body" }), {
							status: 400,
							headers: { "Content-Type": "application/json" },
						});
					}

					const messages = payload.messages;
					const dataField = payload.data;
					const conversationFromData =
						typeof dataField === "object" &&
						dataField !== null &&
						typeof (dataField as { conversationId?: unknown }).conversationId === "string"
							? (dataField as { conversationId: string }).conversationId
							: undefined;
					const conversationId =
						typeof payload.conversationId === "string"
							? payload.conversationId
							: conversationFromData;

					if (!Array.isArray(messages) || messages.length === 0) {
						return new Response(
							JSON.stringify({ error: "messages must be a non-empty array" }),
							{ status: 400, headers: { "Content-Type": "application/json" } },
						);
					}

					if (!messages.every(isUIMessage)) {
						return new Response(
							JSON.stringify({ error: "messages must contain objects with string id + role plus parts[]" }),
							{ status: 400, headers: { "Content-Type": "application/json" } },
						);
					}

					const conversationRegex = /^[^\s]{1,2048}$/;
					let normalizedConversationId: string | undefined;
					if (typeof conversationId === "string") {
						const trimmed = conversationId.trim();
						if (trimmed.length === 0) {
							return new Response(
								JSON.stringify({ error: "conversationId cannot be blank" }),
								{ status: 400, headers: { "Content-Type": "application/json" } },
							);
						}
						if (!conversationRegex.test(trimmed)) {
							return new Response(
								JSON.stringify({ error: "conversationId format invalid" }),
								{ status: 400, headers: { "Content-Type": "application/json" } },
							);
						}
						normalizedConversationId = trimmed;
					}

					// Per-request overrides sent by the client (persona parameters).
					const requestedModel =
						typeof payload.model === "string" ? payload.model : undefined;
					const systemPrompts = Array.isArray(payload.systemPrompts)
						? (payload.systemPrompts.filter((s) => typeof s === "string") as string[])
						: undefined;
					const tools = resolveTools(payload.availableTools);

					const model = coerceOpenAIModel(
						requestedModel ?? process.env.OPENAI_MODEL,
						{ requester: requesterFingerprint },
					);

					const abortController = new AbortController();
					const timeoutHandle = globalThis.setTimeout(
						() => abortController.abort(),
						STREAM_TIMEOUT_MS,
					);

					try {
						const stream = chat({
							adapter: openaiText(model),
							messages,
							conversationId: normalizedConversationId,
							abortController,
							...(systemPrompts && systemPrompts.length > 0 ? { systemPrompts } : {}),
							...(tools ? { tools } : {}),
						});

						logAssistantEvent({
							level: "info",
							conversationId: normalizedConversationId ?? null,
							requester: requesterFingerprint,
							model,
							toolCount: tools?.length ?? 0,
						});

						return toServerSentEventsResponse(stream, { abortController });
					} finally {
						globalThis.clearTimeout(timeoutHandle);
					}
				} catch (error) {
					logAssistantEvent({
						level: "error",
						error: error instanceof Error ? error.message : "unknown_failure",
						requester: requesterFingerprint,
						stack: error instanceof Error ? error.stack : undefined,
					});

					return new Response(
						JSON.stringify({ error: error instanceof Error ? error.message : "An error occurred" }),
						{ status: 500, headers: { "Content-Type": "application/json" } },
					);
				}
			},
		},
	},
});
