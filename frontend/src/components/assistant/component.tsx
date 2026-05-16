import type { UIMessage } from "@tanstack/ai-client";
import {
	Bot,
	CircleAlertIcon,
	Maximize2,
	Minimize2,
	Plus,
	Send,
	Settings,
	Square,
	Trash2,
	X,
} from "lucide-react";
import { useCallback, useState } from "react";
import { Button } from "#/components/ui/button";
import {
	Card,
	CardFrame,
	CardFrameAction,
	CardFrameDescription,
	CardFrameFooter,
	CardFrameHeader,
	CardFrameTitle,
	CardPanel,
} from "#/components/ui/card";
import { Field } from "#/components/ui/field";
import { AnimatePresence, Flex } from "#/components/ui/flex";
import { Form } from "#/components/ui/form";
import { Input } from "#/components/ui/input";
import { Tabs } from "#/components/ui/tabs";
import { cn } from "@/lib/utils";
import { MessageFeed } from "./panels/messages";
import { SettingsPanel } from "./panels/settings";
import { usePageContext } from "./use-page-context";
import { useSession } from "./use-session";
import { useTeamChat } from "./use-team-chat";
import { Header } from "./header";

type Mode = "closed" | "mini" | "full";

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
		updatePersona,
		addPersona,
		removePersona,
		setWindowSize,
	} = useSession();

	const input = useInput();
	const { capture } = usePageContext();
	const [tab, setTab] = useState<string>("chat");

	const { send, stop, status, streamingPersonaId } = useTeamChat(
		session,
		appendMessages,
	);
	const busy = status === "running";

	const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
		e.preventDefault();
		if (!input.value.trim() || busy) return;
		const msg = buildUserMessage(input.value.trim(), capture());
		input.clear();
		send(msg);
	};

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
				isMini && "bottom-6 right-6 w-80 max-h-[70vh]",
				isFull && "inset-4",
			)}
		>
			<CardFrame
				className={cn(
					"flex-1 min-h-0 shadow-2xl",
					isClosed && "bg-primary text-primary-foreground",
				)}
			>
				<Header />
				{!isClosed && (
					<Card>
						<CardPanel>
							<Flex className="h-full min-h-0">
								{isFull && (
									<Flex.Column className="w-52 shrink-0 border-r">
										<Flex.Row
											align="center"
											justify="between"
											className="px-3 py-2"
										>
											<span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
												Sessions
											</span>
											<Button
												size="icon-xs"
												variant="ghost"
												onClick={createSession}
												aria-label="New session"
											>
												<Plus />
											</Button>
										</Flex.Row>
										<Flex.Column className="flex-1 overflow-y-auto">
											<AnimatePresence initial={false}>
												{sessions.map((s) => (
													<Flex.Row
														key={s.id}
														align="center"
														gap={1}
														appear="slideRight"
														className="group"
														layout
													>
														<Button
															variant={
																s.id === session.id ? "secondary" : "ghost"
															}
															className="flex-1 justify-start truncate"
															onClick={() => setActive(s.id)}
														>
															{s.title}
														</Button>
														{sessions.length > 1 && (
															<Button
																size="icon-xs"
																variant="ghost"
																onClick={() => deleteSession(s.id)}
																aria-label="Delete session"
															>
																<Trash2 />
															</Button>
														)}
													</Flex.Row>
												))}
											</AnimatePresence>
										</Flex.Column>
									</Flex.Column>
								)}

								<Flex.Column className="flex-1 min-w-0">
									{isFull && (
										<Flex.Row
											align="center"
											justify="between"
											className="px-4 py-2 border-b"
										>
											<Tabs value={tab} onValueChange={setTab}>
												<Tabs.List>
													<Tabs.Tab value="chat">Chat</Tabs.Tab>
													<Tabs.Tab value="settings">
														<Settings />
														Team
													</Tabs.Tab>
												</Tabs.List>
											</Tabs>
										</Flex.Row>
									)}

									<AnimatePresence mode="wait" initial={false}>
										{(isMini || tab === "chat") && (
											<Flex.Column
												key="chat"
												appear="fadeUp"
												className="flex-1 min-h-0"
											>
												<MessageFeed
													messages={session.messages}
													streamingPersonaId={streamingPersonaId}
													isSubmitted={busy}
													compact={isMini}
												/>

												<Form
													onSubmit={handleSubmit}
													className="flex gap-2 border-t p-3"
												>
													<Field className="flex-1">
														<Field.Label className="sr-only">
															Message
														</Field.Label>
														<Input
															value={input.value}
															onChange={(e) => input.setValue(e.target.value)}
															placeholder={
																session.personas.length > 1 && !isMini
																	? `Message your team (${session.personas.map((p) => p.name).join(", ")})…`
																	: "Message…"
															}
															disabled={busy}
														/>
													</Field>
													<AnimatePresence mode="wait" initial={false}>
														{busy ? (
															<Flex key="stop" appear="scaleIn">
																<Button
																	type="button"
																	variant="outline"
																	onClick={stop}
																>
																	<Square />
																</Button>
															</Flex>
														) : (
															<Flex key="send" appear="scaleIn">
																<Button
																	type="submit"
																	disabled={!input.value.trim()}
																>
																	<Send />
																</Button>
															</Flex>
														)}
													</AnimatePresence>
												</Form>
											</Flex.Column>
										)}

										{isFull && tab === "settings" && (
											<Flex
												key="settings"
												appear="fadeUp"
												className="flex-1 min-h-0 overflow-y-auto"
											>
												<SettingsPanel
													session={session}
													onUpdatePersona={updatePersona}
													onAddPersona={addPersona}
													onRemovePersona={removePersona}
													onWindowSizeChange={setWindowSize}
												/>
											</Flex>
										)}
									</AnimatePresence>
								</Flex.Column>
							</Flex>
						</CardPanel>
					</Card>
				)}

				{isFull && (
					<CardFrameFooter>
						<Flex.Row gap={1} className="text-muted-foreground text-xs">
							<CircleAlertIcon className="size-3 h-lh shrink-0" />
							<p>Responses may be incomplete while the team is streaming.</p>
						</Flex.Row>
					</CardFrameFooter>
				)}
			</CardFrame>
		</Flex>
	);
}
