import { afterEach, describe, expect, it } from "vitest";
import { assistantContextBridge } from "./assistant-context-bridge";

afterEach(() => {
	assistantContextBridge.unpublish("test_key");
	assistantContextBridge.unpublish("other_key");
});

describe("assistantContextBridge.snapshot", () => {
	it("returns a stable array reference until entries change", () => {
		const first = assistantContextBridge.snapshot();

		assistantContextBridge.publish({
			key: "test_key",
			label: "Test",
			value: "hello",
		});

		const afterPublish = assistantContextBridge.snapshot();
		expect(afterPublish).not.toBe(first);
		expect(afterPublish).toHaveLength(1);

		const afterRepeatPublish = assistantContextBridge.snapshot();
		assistantContextBridge.publish({
			key: "test_key",
			label: "Test",
			value: "hello",
		});

		expect(assistantContextBridge.snapshot()).toBe(afterRepeatPublish);
	});

	it("rebuilds snapshot when entry value changes", () => {
		assistantContextBridge.publish({
			key: "test_key",
			label: "Test",
			value: "v1",
		});

		const before = assistantContextBridge.snapshot();

		assistantContextBridge.publish({
			key: "test_key",
			label: "Test",
			value: "v2",
		});

		expect(assistantContextBridge.snapshot()).not.toBe(before);
		expect(assistantContextBridge.snapshot()[0]?.value).toBe("v2");
	});
});
