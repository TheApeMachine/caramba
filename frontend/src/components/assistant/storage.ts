import type { Session } from "./types";
import { DEFAULT_PERSONA, DEFAULT_WINDOW_SIZE } from "./types";

const KEY = "caramba:assistant:sessions";
const ACTIVE_KEY = "caramba:assistant:active";

export function loadSessions(): Session[] {
	try {
		const raw = localStorage.getItem(KEY);
		return raw ? (JSON.parse(raw) as Session[]) : [];
	} catch {
		return [];
	}
}

export function saveSessions(sessions: Session[]): void {
	localStorage.setItem(KEY, JSON.stringify(sessions));
}

export function loadActiveId(): string | null {
	return localStorage.getItem(ACTIVE_KEY);
}

export function saveActiveId(id: string): void {
	localStorage.setItem(ACTIVE_KEY, id);
}

export function newSession(): Session {
	return {
		id: crypto.randomUUID(),
		title: "New conversation",
		createdAt: Date.now(),
		messages: [],
		personas: [{ ...DEFAULT_PERSONA }],
		windowSize: DEFAULT_WINDOW_SIZE,
	};
}

export function deriveTitle(session: Session): string {
	for (const msg of session.messages) {
		if (msg.role !== "user") continue;
		for (const part of msg.parts) {
			if (part.type === "text" && part.content.trim()) {
				return part.content.trim().slice(0, 52);
			}
		}
	}
	return "New conversation";
}

/*
windowedMessages returns the first message (pinned) plus the last `windowSize`
messages, deduplicating the pinned entry so it's never counted twice.
*/
export function windowedMessages(session: Session): Session["messages"] {
	const { messages, windowSize } = session;
	if (messages.length === 0) return [];

	const [pinned, ...rest] = messages;
	const window = rest.slice(-windowSize);

	return [pinned, ...window];
}
