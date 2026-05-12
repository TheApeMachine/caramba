import { EventType } from "@tanstack/ai";
import type { UIMessage } from "@tanstack/ai-client";
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

/*
executeClientTool looks up a tool by name in the registered client tools and
runs its execute function, returning a serialised result string.
*/
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

/*
runTurn sends the current thread to a single persona and collects its full
response, including any tool call / result round-trips, into UIMessages.
*/
async function runTurn(
	persona: Persona,
	thread: UIMessage[],
	session: Session,
	abortSignal: AbortSignal,
): Promise<UIMessage[]> {
	const windowed = windowedMessages({ ...session, messages: thread });
	const adapter = fetchServerSentEvents("/api/assistant");
	const generated: UIMessage[] = [];
	const msgId = crypto.randomUUID();

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
	let currentToolCallId: string | null = null;
	let currentToolName: string | null = null;
	let currentToolArgs = "";

	for await (const chunk of stream) {
		if (abortSignal.aborted) break;

		switch (chunk.type) {
			case EventType.TEXT_MESSAGE_CONTENT:
				textContent += chunk.delta;
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

				generated.push({
					id: crypto.randomUUID(),
					role: "assistant",
					parts: [
						{ type: "tool-call", id: toolCallId, name: toolName, arguments: currentToolArgs, state: "input-complete" },
						{ type: "tool-result", toolCallId, content: result, state: "complete" },
					],
					createdAt: new Date(),
				});

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

	if (textContent.trim()) {
		generated.push({
			id: msgId,
			role: "assistant",
			parts: [{ type: "text", content: `**${persona.name}**: ${textContent}` }],
			createdAt: new Date(),
		});
	}

	return generated;
}

export function useTeamChat(session: Session, onMessages: (msgs: UIMessage[]) => void) {
	const [status, setStatus] = useState<TeamChatStatus>("idle");
	const [streamingPersonaId, setStreamingPersonaId] = useState<string | null>(null);
	const abortRef = useRef<AbortController | null>(null);

	const send = useCallback(
		async (userMessage: UIMessage) => {
			if (status === "running") return;

			abortRef.current = new AbortController();
			const { signal } = abortRef.current;

			setStatus("running");

			try {
				let thread: UIMessage[] = [...session.messages, userMessage];
				onMessages([userMessage]);

				for (const persona of shuffle(session.personas)) {
					if (signal.aborted) break;
					setStreamingPersonaId(persona.id);

					const newMsgs = await runTurn(persona, thread, session, signal);
					if (newMsgs.length > 0) {
						thread = [...thread, ...newMsgs];
						onMessages(newMsgs);
					}
				}
			} catch (err) {
				if (!(err instanceof DOMException && err.name === "AbortError")) {
					setStatus("error");
					return;
				}
			} finally {
				setStreamingPersonaId(null);
				setStatus("idle");
			}
		},
		[status, session, onMessages],
	);

	const stop = useCallback(() => {
		abortRef.current?.abort();
	}, []);

	return { send, stop, status, streamingPersonaId };
}
