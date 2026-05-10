// components/Chat.tsx

import type { MessagePart } from "@tanstack/ai-client";
import { fetchServerSentEvents, useChat } from "@tanstack/ai-react";
import { useState } from "react";

/*
messagePartsWithStableKeys derives list keys aligned with TanStack stream behavior:
tool parts use canonical ids; each text/thinking occurrence gets a stable slot key
without using the array index as the React key.
*/
function messagePartsWithStableKeys(
	messageId: string,
	parts: ReadonlyArray<MessagePart>,
): Array<{ stableKey: string; part: MessagePart }> {
	let textOccurrence = 0;
	let thinkingFallbackOccurrence = 0;
	const otherOccurrenceByType = new Map<string, number>();
	const keyed: Array<{ stableKey: string; part: MessagePart }> = [];

	for (const part of parts) {
		switch (part.type) {
			case "tool-call": {
				keyed.push({
					stableKey: `${messageId}:tool-call:${part.id}`,
					part,
				});
				break;
			}

			case "tool-result": {
				keyed.push({
					stableKey: `${messageId}:tool-result:${part.toolCallId}`,
					part,
				});
				break;
			}

			case "text": {
				keyed.push({
					stableKey: `${messageId}:text:${textOccurrence}`,
					part,
				});
				textOccurrence += 1;
				break;
			}

			case "thinking": {
				const thinkingPart = part as MessagePart & { stepId?: string };
				const hasStepId =
					typeof thinkingPart.stepId === "string" && thinkingPart.stepId !== "";

				const fallbackId = `slot-${thinkingFallbackOccurrence}`;
				thinkingFallbackOccurrence += 1;
				const discriminator = hasStepId ? thinkingPart.stepId : fallbackId;

				keyed.push({
					stableKey: `${messageId}:thinking:${discriminator}`,
					part,
				});
				break;
			}

			default: {
				const next = otherOccurrenceByType.get(part.type) ?? 0;
				otherOccurrenceByType.set(part.type, next + 1);
				keyed.push({
					stableKey: `${messageId}:${part.type}:${next}`,
					part,
				});
			}
		}
	}

	return keyed;
}

export function Chat() {
	const [input, setInput] = useState("");

	const { messages, sendMessage, isLoading } = useChat({
		connection: fetchServerSentEvents("/api/assistant"),
	});

	const handleSubmit = (event: React.SubmitEvent<HTMLFormElement>) => {
		event.preventDefault();
		if (input.trim() && !isLoading) {
			sendMessage(input);
			setInput("");
		}
	};

	return (
		<div className="flex flex-col h-screen">
			{/* Messages */}
			<div className="flex-1 overflow-y-auto p-4">
				{messages.map((message) => (
					<div
						key={message.id}
						className={`mb-4 ${
							message.role === "assistant" ? "text-blue-600" : "text-gray-800"
						}`}
					>
						<div className="font-semibold mb-1">
							{message.role === "assistant" ? "Assistant" : "You"}
						</div>
						<div>
							{messagePartsWithStableKeys(message.id, message.parts).map(
								({ stableKey, part }) => {
									if (part.type === "thinking") {
										return (
											<div
												key={stableKey}
												className="text-sm text-gray-500 italic mb-2"
											>
												💭 Thinking: {part.content}
											</div>
										);
									}

									if (part.type === "text") {
										return <div key={stableKey}>{part.content}</div>;
									}

									if (part.type === "tool-call") {
										return (
											<div
												key={stableKey}
												className="text-xs rounded border border-blue-100 bg-blue-50 text-blue-900 p-3 mb-2"
											>
												<p className="font-semibold">Tool • {part.name}</p>
												<pre className="mt-1 whitespace-pre-wrap wrap-break-word">
													{part.arguments}
												</pre>
											</div>
										);
									}

									if (part.type === "tool-result") {
										return (
											<div
												key={stableKey}
												className="text-xs rounded border border-emerald-100 bg-emerald-50 text-emerald-900 p-3 mb-2"
											>
												<p className="font-semibold">Tool result</p>
												<p className="text-[11px] uppercase tracking-wide text-emerald-700">
													{part.state}
												</p>
												<pre className="mt-1 whitespace-pre-wrap wrap-break-word">
													{part.error ?? part.content}
												</pre>
											</div>
										);
									}

									return null;
								},
							)}
						</div>
					</div>
				))}
			</div>

			{/* Input */}
			<form onSubmit={handleSubmit} className="p-4 border-t">
				<div className="flex gap-2">
					<label htmlFor="assistant-message-input" className="sr-only">
						Message input
					</label>
					<input
						id="assistant-message-input"
						type="text"
						value={input}
						onChange={(e) => setInput(e.target.value)}
						placeholder="Type a message..."
						className="flex-1 px-4 py-2 border rounded-lg"
						disabled={isLoading}
					/>
					<button
						type="submit"
						disabled={!input.trim() || isLoading}
						className="px-6 py-2 bg-blue-600 text-white rounded-lg disabled:opacity-50"
					>
						Send
					</button>
				</div>
			</form>
		</div>
	);
}
