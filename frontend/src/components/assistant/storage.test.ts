import { describe, expect, it } from "vitest";
import { estimateMessageTokens, windowedMessages } from "./storage";
import type { Session, UIMessage } from "./types";

const message = (
	id: string,
	content: string,
	role: UIMessage["role"] = "user",
): UIMessage => ({
	id,
	role,
	parts: [{ type: "text", content }],
	createdAt: new Date(),
});

describe("windowedMessages", () => {
	it("keeps pinned message and respects token budget", () => {
		const session: Session = {
			id: "session",
			scope: "personal",
			title: "Test",
			createdAt: Date.now(),
			personas: [],
			windowSize: 1,
			messages: [
				message("pinned", "system prompt anchor"),
				message("a", "a".repeat(1200)),
				message("b", "b".repeat(1200)),
				message("c", "recent short"),
			],
		};

		const windowed = windowedMessages(session);

		expect(windowed[0]?.id).toBe("pinned");
		expect(windowed.at(-1)?.id).toBe("c");
		expect(windowed.length).toBeLessThan(session.messages.length);
	});
});

describe("estimateMessageTokens", () => {
	it("estimates from text part length", () => {
		const tokens = estimateMessageTokens(message("m", "abcd"));
		expect(tokens).toBeGreaterThan(0);
	});
});
