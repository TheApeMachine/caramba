import type { UIMessage } from "@tanstack/ai-client";
import { Bot, Maximize2, Minimize2, Plus, Send, Settings, Square, Trash2, X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useState } from "react";
import { Button } from "#/components/ui/button";
import { Flex } from "#/components/ui/flex";
import { Input } from "#/components/ui/input";
import { Tabs } from "#/components/ui/tabs";
import { cn } from "@/lib/utils";
import { MessageFeed } from "./panels/messages";
import { SettingsPanel } from "./panels/settings";
import { usePageContext } from "./use-page-context";
import { useSession } from "./use-session";
import { useTeamChat } from "./use-team-chat";

const panelSpring = { type: "spring", stiffness: 380, damping: 32, mass: 0.7 } as const;
const fadeFast = { duration: 0.18, ease: [0.22, 1, 0.36, 1] } as const;

function useInput() {
	const [value, setValue] = useState("");

	const clear = useCallback(() => setValue(""), []);

	return { value, setValue, clear };
}

/*
buildUserMessage prepends a page context block when available so agents always
know what the user is looking at without the user having to explain it.
*/
function buildUserMessage(text: string, pageContext: string): UIMessage {
	const content = pageContext ? `${pageContext}\n\n---\n\n${text}` : text;
	return {
		id: crypto.randomUUID(),
		role: "user",
		parts: [{ type: "text", content }],
		createdAt: new Date(),
	};
}

/* ── Mini panel (compact, bottom-right) ──────────────────────────────────── */

function MiniPanel({
	onExpand,
	onClose,
}: {
	onExpand: () => void;
	onClose: () => void;
}) {
	const { session, appendMessages } = useSession();
	const input = useInput();
	const { capture } = usePageContext();

	const { send, stop, status, streamingPersonaId } = useTeamChat(session, appendMessages);

	const busy = status === "running";

	const handleSubmit = (e: React.SubmitEvent<HTMLFormElement>) => {
		e.preventDefault();
		if (!input.value.trim() || busy) return;
		const msg = buildUserMessage(input.value.trim(), capture());
		input.clear();
		send(msg);
	};

	return (
		<div className="flex flex-col w-80 rounded-2xl border bg-background shadow-2xl overflow-hidden">
			<div className="flex items-center justify-between px-4 py-3 border-b bg-muted/40 shrink-0">
				<div className="flex items-center gap-2">
					<Bot className="size-4 text-muted-foreground" />
					<span className="text-sm font-medium">
						{session.personas.length > 1
							? `Research team (${session.personas.length})`
							: (session.personas[0]?.name ?? "Assistant")}
					</span>
				</div>
				<div className="flex items-center gap-1">
					<Button size="icon-xs" variant="ghost" onClick={onExpand} aria-label="Expand">
						<Maximize2 />
					</Button>
					<Button size="icon-xs" variant="ghost" onClick={onClose} aria-label="Close">
						<X />
					</Button>
				</div>
			</div>

			<MessageFeed
				messages={session.messages}
				streamingPersonaId={streamingPersonaId}
				isSubmitted={busy}
				compact
			/>

			<form onSubmit={handleSubmit} className="flex gap-2 px-3 py-3 border-t shrink-0">
				<Input
					value={input.value}
					onChange={(e) => input.setValue(e.target.value)}
					placeholder="Message…"
					disabled={busy}
					className="h-8 text-sm"
				/>
				{busy ? (
					<Button type="button" size="icon-sm" variant="outline" onClick={stop}>
						<Square />
					</Button>
				) : (
					<Button type="submit" size="icon-sm" disabled={!input.value.trim()}>
						<Send />
					</Button>
				)}
			</form>
		</div>
	);
}

/* ── Full overlay ─────────────────────────────────────────────────────────── */

function FullOverlay({ onCollapse }: { onCollapse: () => void }) {
	const {
		sessions,
		session,
		setActive,
		createSession,
		deleteSession,
		appendMessages,
		updatePersona,
		addPersona,
		removePersona,
		setWindowSize,
	} = useSession();

	const input = useInput();
	const { capture } = usePageContext();
	const [tab, setTab] = useState<string>("chat");

	const { send, stop, status, streamingPersonaId } = useTeamChat(session, appendMessages);

	const busy = status === "running";

	const handleSubmit = (e: React.SubmitEvent<HTMLFormElement>) => {
		e.preventDefault();
		if (!input.value.trim() || busy) return;
		const msg = buildUserMessage(input.value.trim(), capture());
		input.clear();
		send(msg);
	};

	return (
		<div className="fixed inset-4 z-50 flex rounded-2xl border bg-background shadow-2xl overflow-hidden">
			{/* Sidebar — session list */}
			<aside className="w-52 shrink-0 border-r flex flex-col bg-muted/20">
				<div className="flex items-center justify-between px-3 py-3 border-b">
					<span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
						Sessions
					</span>
					<Button size="icon-xs" variant="ghost" onClick={createSession} aria-label="New session">
						<Plus />
					</Button>
				</div>
				<div className="flex-1 overflow-y-auto py-1">
					{sessions.map((s) => (
						<div key={s.id} className="group flex items-center gap-1 mx-1">
							<button
								type="button"
								className={cn(
									"flex-1 flex items-center gap-1 px-3 py-2 rounded-lg text-left text-xs truncate",
									s.id === session.id
										? "bg-accent text-accent-foreground"
										: "hover:bg-accent/50 text-muted-foreground",
								)}
								onClick={() => setActive(s.id)}
							>
								{s.title}
							</button>
							{sessions.length > 1 && (
								<Button
									size="icon-xs"
									variant="ghost"
									className="opacity-0 group-hover:opacity-100 shrink-0"
									onClick={() => deleteSession(s.id)}
								>
									<Trash2 />
								</Button>
							)}
						</div>
					))}
				</div>
			</aside>

			{/* Main area */}
			<div className="flex flex-col flex-1 min-w-0">
				{/* Header */}
				<div className="flex items-center justify-between px-4 py-3 border-b shrink-0">
					<Tabs value={tab} onValueChange={setTab}>
						<Tabs.List>
							<Tabs.Tab value="chat">Chat</Tabs.Tab>
							<Tabs.Tab value="settings">
								<Settings className="size-3.5 mr-1" />
								Team
							</Tabs.Tab>
						</Tabs.List>
					</Tabs>
					<Button size="icon-xs" variant="ghost" onClick={onCollapse} aria-label="Collapse">
						<Minimize2 />
					</Button>
				</div>

				{tab === "chat" && (
					<>
						<MessageFeed
							messages={session.messages}
							streamingPersonaId={streamingPersonaId}
							isSubmitted={busy}
						/>

						<form
							onSubmit={handleSubmit}
							className="flex gap-2 px-4 py-4 border-t shrink-0"
						>
							<Input
								value={input.value}
								onChange={(e) => input.setValue(e.target.value)}
								placeholder={
									session.personas.length > 1
										? `Message your team (${session.personas.map((p) => p.name).join(", ")})…`
										: "Message…"
								}
								disabled={busy}
								className="text-sm"
							/>
							{busy ? (
								<Button type="button" variant="outline" onClick={stop}>
									<Square />
									Stop
								</Button>
							) : (
								<Button type="submit" disabled={!input.value.trim()}>
									<Send />
									Send
								</Button>
							)}
						</form>
					</>
				)}

				{tab === "settings" && (
					<SettingsPanel
						session={session}
						onUpdatePersona={updatePersona}
						onAddPersona={addPersona}
						onRemovePersona={removePersona}
						onWindowSizeChange={setWindowSize}
					/>
				)}
			</div>
		</div>
	);
}

/* ── Root ─────────────────────────────────────────────────────────────────── */

export function Assistant() {
	const [state, setState] = useState<"closed" | "mini" | "full">("closed");

	return (
		<>
			{state === "full" && (
				<FullOverlay onCollapse={() => setState("mini")} />
			)}

			<div className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-3">
				{state === "mini" && (
					<div
						className="transition-all duration-300 origin-bottom-right"
						style={{ maxHeight: "520px" }}
					>
						<MiniPanel
							onExpand={() => setState("full")}
							onClose={() => setState("closed")}
						/>
					</div>
				)}

				<Button
					size="icon-xl"
					variant={state !== "closed" ? "outline" : "default"}
					className="rounded-full shadow-lg"
					onClick={() => setState((s) => s === "closed" ? "mini" : "closed")}
					aria-label="Toggle assistant"
				>
					<Bot />
				</Button>
			</div>
		</>
	);
}
