import type { MessagePart } from "@tanstack/ai-client";
import type { Session, UIMessage } from "./types";

const TOKENS_PER_MESSAGE_SLOT = 400;

/*
estimateMessageTokens approximates token weight for context budgeting.
Uses a conservative chars/4 heuristic over all message parts.
*/
export function estimateMessageTokens(message: UIMessage): number {
	const parts = (message.parts ?? []) as MessagePart[];
	const text = parts
		.map((part) => {
			if (part.type === "text") {
				return part.content;
			}

			return JSON.stringify(part);
		})
		.join("\n");

	return Math.max(1, Math.ceil(text.length / 4));
}

/*
windowedMessages keeps the pinned first message and fills the remainder from
the end until the token budget is exhausted. `windowSize` maps to budget as
windowSize * TOKENS_PER_MESSAGE_SLOT (20 slots ≈ 8000 tokens).
*/
export function windowedMessages(session: Session): Session["messages"] {
	const { messages, windowSize } = session;

	if (messages.length === 0) {
		return [];
	}

	const tokenBudget = Math.max(
		TOKENS_PER_MESSAGE_SLOT,
		windowSize * TOKENS_PER_MESSAGE_SLOT,
	);
	const [pinned, ...rest] = messages;
	const tail: UIMessage[] = [];
	let usedTokens = estimateMessageTokens(pinned);

	for (let index = rest.length - 1; index >= 0; index -= 1) {
		const message = rest[index];
		const messageTokens = estimateMessageTokens(message);

		if (usedTokens + messageTokens > tokenBudget && tail.length > 0) {
			break;
		}

		tail.unshift(message);
		usedTokens += messageTokens;
	}

	return [pinned, ...tail];
}
