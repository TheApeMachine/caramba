import type { Session } from "./types";

/*
windowedMessages returns the first message (pinned) plus the last `windowSize`
messages, deduplicating the pinned entry so it's never counted twice. Used by
the chat hook to bound the context sent to the model.
*/
export function windowedMessages(session: Session): Session["messages"] {
	const { messages, windowSize } = session;
	if (messages.length === 0) return [];

	const [pinned, ...rest] = messages;
	const window = rest.slice(-windowSize);

	return [pinned, ...window];
}
