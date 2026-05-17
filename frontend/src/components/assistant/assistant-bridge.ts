import type { Mode } from "./types";

/*
assistantBridge is a singleton that the Assistant component registers its
send + setMode handlers into. External UIs (slash menus, command palettes,
shortcuts) can dispatch a user message into the running assistant session
without having to lift state up.
*/
export type AssistantBridgeAPI = {
	send: (text: string) => void;
	setMode: (mode: Mode) => void;
};

let api: AssistantBridgeAPI | null = null;
const subscribers = new Set<(ready: boolean) => void>();

function notify() {
	for (const subscriber of subscribers) {
		subscriber(api !== null);
	}
}

export const assistantBridge = {
	register(impl: AssistantBridgeAPI) {
		api = impl;
		notify();
	},
	unregister() {
		api = null;
		notify();
	},
	get(): AssistantBridgeAPI | null {
		return api;
	},
	subscribe(callback: (ready: boolean) => void): () => void {
		subscribers.add(callback);
		callback(api !== null);
		return () => {
			subscribers.delete(callback);
		};
	},
};
