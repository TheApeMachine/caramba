import { Settings } from "lucide-react";
import { useState } from "react";
import { Card, CardPanel } from "#/components/ui/card";
import { AnimatePresence, Flex } from "#/components/ui/flex";
import { Tabs } from "#/components/ui/tabs";
import { Composer } from "./composer";
import { MessageFeed } from "./panels/messages";
import { SettingsPanel } from "./panels/settings";
import { Sidebar } from "./sidebar";
import type { Mode, Persona, Session } from "./types";

interface BodyProps {
	mode: Mode;
	session: Session;
	sessions: Session[];
	streamingPersonaId: string | null | undefined;
	busy: boolean;
	inputValue: string;
	onInputChange: (value: string) => void;
	onSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
	onStop: () => void;
	onSelectSession: (id: string) => void;
	onCreateSession: () => void;
	onDeleteSession: (id: string) => void;
	onUpdatePersona: (persona: Persona) => void;
	onAddPersona: (persona: Persona) => void;
	onRemovePersona: (id: string) => void;
	onWindowSizeChange: (size: number) => void;
}

export const Body = ({
	mode,
	session,
	sessions,
	streamingPersonaId,
	busy,
	inputValue,
	onInputChange,
	onSubmit,
	onStop,
	onSelectSession,
	onCreateSession,
	onDeleteSession,
	onUpdatePersona,
	onAddPersona,
	onRemovePersona,
	onWindowSizeChange,
}: BodyProps) => {
	const [tab, setTab] = useState<string>("chat");
	const isMini = mode === "mini";
	const isFull = mode === "full";

	const placeholder =
		session.personas.length > 1 && !isMini
			? `Message your team (${session.personas.map((p) => p.name).join(", ")})…`
			: "Message…";

	return (
		<Card>
			<CardPanel>
				<Flex className="h-full min-h-0">
					{isFull && (
						<Sidebar
							sessions={sessions}
							activeId={session.id}
							onSelect={onSelectSession}
							onCreate={onCreateSession}
							onDelete={onDeleteSession}
						/>
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
									<Composer
										value={inputValue}
										onChange={onInputChange}
										onSubmit={onSubmit}
										onStop={onStop}
										busy={busy}
										placeholder={placeholder}
									/>
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
										onUpdatePersona={onUpdatePersona}
										onAddPersona={onAddPersona}
										onRemovePersona={onRemovePersona}
										onWindowSizeChange={onWindowSizeChange}
									/>
								</Flex>
							)}
						</AnimatePresence>
					</Flex.Column>
				</Flex>
			</CardPanel>
		</Card>
	);
};
