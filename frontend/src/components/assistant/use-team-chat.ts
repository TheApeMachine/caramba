import { EventType } from "@tanstack/ai";
import type { MessagePart, UIMessage } from "@tanstack/ai-client";
import { fetchServerSentEvents } from "@tanstack/ai-react";
import { useCallback, useRef, useState } from "react";
import { paperEditorTools } from "./tools/paper-editor";
import type { Persona, Session } from "./types";
import { windowedMessages } from "./storage";

export type TeamChatStatus = "idle" | "running" | "error";

function shuffle<T>(arr: T[]): T[] {
	const out = [...arr];
	for (let i = out.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[out[i], out[j]] = [out[j], out[i]];
	}
	return out;
}

async function executeClientTool(name: string, args: unknown): Promise<string> {
	const tool = paperEditorTools.find((t) => t.name === name);
	if (!tool?.execute) return JSON.stringify({ error: `Unknown tool: ${name}` });
	try {
		const result = await tool.execute(args as never);
		return JSON.stringify(result);
	} catch (err) {
		return JSON.stringify({ error: err instanceof Error ? err.message : String(err) });
	}
}

interface TurnCallbacks {
	upsertMessage: (msg: UIMessage) => void;
	appendMessages: (msgs: UIMessage[]) => void;
	onReasoningActive: (active: boolean) => void;
}

async function runTurn(
	persona: Persona,
	thread: UIMessage[],
	session: Session,
	abortSignal: AbortSignal,
	cb: TurnCallbacks,
): Promise<UIMessage[]> {
	const windowed = windowedMessages({ ...session, messages: thread });
	const adapter = fetchServerSentEvents("/api/assistant");
	const generated: UIMessage[] = [];
	const streamingMsgId = crypto.randomUUID();

	const toolNames = paperEditorTools.map((t) => t.name);

	const stream = await adapter.connect(
		windowed,
		{
			systemPrompts: [persona.systemPrompt],
			model: persona.model,
			temperature: persona.temperature,
			maxTokens: persona.maxTokens,
			personaName: persona.name,
			availableTools: toolNames,
		},
		abortSignal,
	);

	let textContent = "";
	let reasoningContent = "";
	let hasStreamingMessage = false;
	let currentToolCallId: string | null = null;
	let currentToolName: string | null = null;
	let currentToolArgs = "";

	const emitStreamingMessage = () => {
		const parts: MessagePart[] = [];
		if (reasoningContent) {
			parts.push({ type: "thinking", content: reasoningContent } as MessagePart);
		}
		if (textContent) {
			parts.push({
				type: "text",
				content: `**${persona.name}**: ${textContent}`,
			} as MessagePart);
		}
		if (parts.length === 0) return;
		hasStreamingMessage = true;
		cb.upsertMessage({
			id: streamingMsgId,
			role: "assistant",
			parts,
			createdAt: new Date(),
		});
	};

	for await (const chunk of stream) {
		if (abortSignal.aborted) break;

		switch (chunk.type) {
			case EventType.TEXT_MESSAGE_CONTENT:
				textContent += chunk.delta;
				emitStreamingMessage();
				break;

			case EventType.REASONING_MESSAGE_START:
			case EventType.REASONING_START:
				cb.onReasoningActive(true);
				break;

			case EventType.REASONING_MESSAGE_CONTENT:
				reasoningContent += chunk.delta ?? "";
				cb.onReasoningActive(true);
				emitStreamingMessage();
				break;

			case EventType.REASONING_MESSAGE_END:
			case EventType.REASONING_END:
				cb.onReasoningActive(false);
				break;

			case EventType.TOOL_CALL_START:
				currentToolCallId = chunk.toolCallId ?? crypto.randomUUID();
				currentToolName = chunk.toolCallName ?? null;
				currentToolArgs = "";
				break;

			case EventType.TOOL_CALL_ARGS:
				currentToolArgs += chunk.delta ?? "";
				break;

			case EventType.TOOL_CALL_END: {
				if (!currentToolName || !currentToolCallId) break;
				const toolCallId = currentToolCallId;
				const toolName = currentToolName;
				let parsedArgs: unknown = {};
				try { parsedArgs = JSON.parse(currentToolArgs); } catch { /* use empty */ }

				const result = await executeClientTool(toolName, parsedArgs);

				const toolMsg: UIMessage = {
					id: crypto.randomUUID(),
					role: "assistant",
					parts: [
						{ type: "tool-call", id: toolCallId, name: toolName, arguments: currentToolArgs, state: "input-complete" },
						{ type: "tool-result", toolCallId, content: result, state: "complete" },
					],
					createdAt: new Date(),
				};
				generated.push(toolMsg);
				cb.appendMessages([toolMsg]);

				currentToolCallId = null;
				currentToolName = null;
				currentToolArgs = "";
				break;
			}

			case EventType.RUN_FINISHED:
				break;

			default:
				break;
		}
	}

	cb.onReasoningActive(false);

	if (hasStreamingMessage) {
		const finalParts: MessagePart[] = [];
		if (reasoningContent) {
			finalParts.push({ type: "thinking", content: reasoningContent } as MessagePart);
		}
		if (textContent) {
			finalParts.push({
				type: "text",
				content: `**${persona.name}**: ${textContent}`,
			} as MessagePart);
		}
		generated.push({
			id: streamingMsgId,
			role: "assistant",
			parts: finalParts,
			createdAt: new Date(),
		});
	}

	return generated;
}

export function useTeamChat(
	session: Session,
	appendMessages: (msgs: UIMessage[]) => void,
	upsertMessage: (msg: UIMessage) => void,
) {
	const [status, setStatus] = useState<TeamChatStatus>("idle");
	const [streamingPersonaId, setStreamingPersonaId] = useState<string | null>(null);
	const [reasoningActive, setReasoningActive] = useState(false);
	const abortRef = useRef<AbortController | null>(null);

	const send = useCallback(
		async (userMessage: UIMessage) => {
			if (status === "running") return;

			abortRef.current = new AbortController();
			const { signal } = abortRef.current;

			setStatus("running");

			try {
				let thread: UIMessage[] = [...session.messages, userMessage];
				appendMessages([userMessage]);

				for (const persona of shuffle(session.personas)) {
					if (signal.aborted) break;
					setStreamingPersonaId(persona.id);
					setReasoningActive(false);

					const newMsgs = await runTurn(persona, thread, session, signal, {
						upsertMessage,
						appendMessages,
						onReasoningActive: setReasoningActive,
					});
					if (newMsgs.length > 0) {
						thread = [...thread, ...newMsgs];
					}
				}
			} catch (err) {
				if (!(err instanceof DOMException && err.name === "AbortError")) {
					setStatus("error");
					return;
				}
			} finally {
				setStreamingPersonaId(null);
				setReasoningActive(false);
				setStatus("idle");
			}
		},
		[status, session, appendMessages, upsertMessage],
	);

	const stop = useCallback(() => {
		abortRef.current?.abort();
	}, []);

	return { send, stop, status, streamingPersonaId, reasoningActive };
}
