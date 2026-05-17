import type { UIMessage } from "@tanstack/ai-client";
import { useCallback, useEffect, useState } from "react";
import { CardFrame } from "#/components/ui/card";
import { Flex } from "#/components/ui/flex";
import { cn } from "@/lib/utils";
import { assistantBridge } from "./assistant-bridge";
import { Body } from "./body";
import { Footer } from "./footer";
import { Header } from "./header";
import type { Mode } from "./types";
import { usePageContext } from "./use-page-context";
import { useSession } from "./use-session";
import { useTeamChat } from "./use-team-chat";

function useInput() {
	const [value, setValue] = useState("");
	const clear = useCallback(() => setValue(""), []);
	return { value, setValue, clear };
}

function buildUserMessage(text: string, pageContext: string): UIMessage {
	const content = pageContext ? `${pageContext}\n\n---\n\n${text}` : text;
	return {
		id: crypto.randomUUID(),
		role: "user",
		parts: [{ type: "text", content }],
		createdAt: new Date(),
	};
}

export function Assistant() {
	const [mode, setMode] = useState<Mode>("closed");

	const {
		sessions,
		session,
		setActive,
		createSession,
		deleteSession,
		appendMessages,
		upsertMessage,
		updatePersona,
		addPersona,
		removePersona,
		setWindowSize,
	} = useSession();

	const input = useInput();
	const { capture } = usePageContext();

	const { send, stop, status, streamingPersonaId, reasoningActive } =
		useTeamChat(session, appendMessages, upsertMessage);
	const busy = status === "running";

	const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
		e.preventDefault();
		if (!input.value.trim() || busy) return;
		const msg = buildUserMessage(input.value.trim(), capture());
		input.clear();
		send(msg);
	};

	useEffect(() => {
		assistantBridge.register({
			send: (text: string) => {
				const trimmed = text.trim();
				if (!trimmed) return;
				send(buildUserMessage(trimmed, capture()));
			},
			setMode,
		});
		return () => assistantBridge.unregister();
	}, [send, capture]);

	const isClosed = mode === "closed";
	const isMini = mode === "mini";
	const isFull = mode === "full";

	const teamName = isMini
		? "Assistant"
		: session.personas.length > 1
			? `Research team (${session.personas.length})`
			: (session.personas[0]?.name ?? "Assistant");

	return (
		<Flex
			layout
			transition={{ type: "spring", stiffness: 320, damping: 32, mass: 0.7 }}
			className={cn(
				"fixed z-50",
				isClosed && "bottom-6 right-6 size-14",
				isMini && "bottom-6 right-6 w-80 h-[70vh]",
				isFull && "inset-4",
			)}
		>
			<CardFrame
				className={cn(
					"w-full flex-1 min-h-0 overflow-hidden shadow-2xl",
					isClosed && "bg-primary text-primary-foreground",
				)}
			>
				<Header mode={mode} setMode={setMode} teamName={teamName} />
				{!isClosed && (
					<Body
						mode={mode}
						session={session}
						sessions={sessions}
						streamingPersonaId={streamingPersonaId}
						reasoningActive={reasoningActive}
						busy={busy}
						inputValue={input.value}
						onInputChange={input.setValue}
						onSubmit={handleSubmit}
						onStop={stop}
						onSelectSession={setActive}
						onCreateSession={createSession}
						onDeleteSession={deleteSession}
						onUpdatePersona={updatePersona}
						onAddPersona={addPersona}
						onRemovePersona={removePersona}
						onWindowSizeChange={setWindowSize}
					/>
				)}
				{isFull && <Footer />}
			</CardFrame>
		</Flex>
	);
}
