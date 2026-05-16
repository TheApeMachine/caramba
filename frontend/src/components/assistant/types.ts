import type { UIMessage as TanstackUIMessage } from "@tanstack/ai-client";

export type UIMessage = TanstackUIMessage & {
	personaId?: string;
	personaName?: string;
};

export type Mode = "closed" | "mini" | "full";

export type PersonaScope = "global" | "team" | "personal";
export type SessionScope = "team" | "personal";
export type AdapterType = "openai" | "ollama" | "openai-compat";

export type Persona = {
	id: string;
	scope: PersonaScope;
	name: string;
	systemPrompt: string;
	model: string;
	temperature: number;
	maxTokens: number;
	adapterType: AdapterType;
	endpointUrl: string;
};

export type Session = {
	id: string;
	scope: SessionScope;
	title: string;
	createdAt: number;
	messages: UIMessage[];
	personas: Persona[];
	windowSize: number;
};

export type { TanstackUIMessage };

export const DEFAULT_PERSONA: Persona = {
	id: "00000000-0000-0000-0000-000000000001",
	scope: "global",
	name: "Assistant",
	systemPrompt:
		"You are a helpful research assistant. You can search arXiv for papers when relevant.",
	model: "gpt-5.4-mini",
	temperature: 0.7,
	maxTokens: 2048,
	adapterType: "openai",
	endpointUrl: "",
};

export const AVAILABLE_MODELS = [
	// OpenAI
	{ id: "gpt-5.5", label: "GPT 5.5", provider: "openai" },
	{ id: "gpt-5.4-mini", label: "GPT 5.4 mini", provider: "openai" },
	// Anthropic
	{ id: "claude-opus-4-7", label: "Claude Opus 4.7", provider: "anthropic" },
	{
		id: "claude-sonnet-4-6",
		label: "Claude Sonnet 4.6",
		provider: "anthropic",
	},
	// Google
	{ id: "gemini-3.1-pro", label: "Gemini 3.1 Pro", provider: "google" },
	{ id: "gemini-3.1-flash", label: "Gemini 3.1 Flash", provider: "google" },
	// xAI
	{ id: "grok-4.3", label: "Grok 4.3", provider: "xai" },
	{ id: "grok-4.3-mini", label: "Grok 4.3 mini", provider: "xai" },
] as const;

export type ModelId = (typeof AVAILABLE_MODELS)[number]["id"];
export type Provider = (typeof AVAILABLE_MODELS)[number]["provider"];

export const DEFAULT_WINDOW_SIZE = 20;
