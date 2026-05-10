import { chat, toServerSentEventsResponse } from "@tanstack/ai";
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

	while (
		index < timestamps.length &&
		nowMs - timestamps[index] > RATE_LIMIT_WINDOW_MS
	) {
		index += 1;
	}

	timestamps.splice(0, index);
	timestamps.push(nowMs);

	return timestamps.length;
}

function getClientFingerprint(request: Request) {
	const forwarded = request.headers.get("x-forwarded-for");

	if (forwarded) {
		return forwarded.split(",")[0]?.trim() ?? "unknown-forwarded";
	}

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
	opts: {
		requester: string;
	},
): SupportedOpenAIModel {
	const fallback: SupportedOpenAIModel = "gpt-4o-mini";
	const trimmed = requestedRaw?.trim();

	if (!trimmed) return fallback;

	const catalogue = OPENAI_CHAT_MODELS as ReadonlyArray<string>;

	if (!catalogue.includes(trimmed)) {
		logAssistantEvent({
			level: "warn",
			message: `Unknown OPENAI_MODEL "${trimmed}", falling back to ${fallback}`,
			requester: opts.requester,
		});

		return fallback;
	}

	return trimmed as SupportedOpenAIModel;
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
						return new Response(
							JSON.stringify({ error: "Too many requests" }),
							{
								status: 429,
								headers: { "Content-Type": "application/json" },
							},
						);
					}

					if (!process.env.OPENAI_API_KEY) {
						return new Response(
							JSON.stringify({
								error: "OPENAI_API_KEY not configured",
							}),
							{
								status: 503,
								headers: { "Content-Type": "application/json" },
							},
						);
					}

					let payload: Record<string, unknown>;

					try {
						payload = (await request.json()) as Record<string, unknown>;
					} catch {
						return new Response(
							JSON.stringify({ error: "Invalid JSON body" }),
							{
								status: 400,
								headers: { "Content-Type": "application/json" },
							},
						);
					}

					const messages = payload.messages;
					const dataField = payload.data;
					const conversationFromData =
						typeof dataField === "object" &&
						dataField !== null &&
						typeof (dataField as { conversationId?: unknown })
							.conversationId === "string"
							? (dataField as { conversationId: string }).conversationId
							: undefined;
					const conversationId =
						typeof payload.conversationId === "string"
							? payload.conversationId
							: conversationFromData;

					if (!Array.isArray(messages) || messages.length === 0) {
						return new Response(
							JSON.stringify({
								error: "messages must be a non-empty array",
							}),
							{
								status: 400,
								headers: { "Content-Type": "application/json" },
							},
						);
					}

					if (!messages.every(isUIMessage)) {
						return new Response(
							JSON.stringify({
								error:
									"messages must contain objects with string id + role plus parts[]",
							}),
							{
								status: 400,
								headers: { "Content-Type": "application/json" },
							},
						);
					}

					const conversationRegex = /^[^\s]{1,2048}$/;
					let normalizedConversationId: string | undefined;

					if (typeof conversationId === "string") {
						const trimmed = conversationId.trim();

						if (trimmed.length === 0) {
							return new Response(
								JSON.stringify({ error: "conversationId cannot be blank" }),
								{
									status: 400,
									headers: { "Content-Type": "application/json" },
								},
							);
						}

						if (!conversationRegex.test(trimmed)) {
							return new Response(
								JSON.stringify({ error: "conversationId format invalid" }),
								{
									status: 400,
									headers: { "Content-Type": "application/json" },
								},
							);
						}

						normalizedConversationId = trimmed;
					}

					const model = coerceOpenAIModel(process.env.OPENAI_MODEL, {
						requester: requesterFingerprint,
					});

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
						});

						logAssistantEvent({
							level: "info",
							conversationId: normalizedConversationId ?? null,
							requester: requesterFingerprint,
							model,
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
						JSON.stringify({
							error:
								error instanceof Error ? error.message : "An error occurred",
						}),
						{
							status: 500,
							headers: { "Content-Type": "application/json" },
						},
					);
				}
			},
		},
	},
});
