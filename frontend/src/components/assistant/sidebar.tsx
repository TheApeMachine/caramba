import { Plus, Trash2 } from "lucide-react";
import { Button } from "#/components/ui/button";
import { AnimatePresence, Flex } from "#/components/ui/flex";
import type { Session } from "./types";

interface SidebarProps {
	sessions: Session[];
	activeId: string;
	onSelect: (id: string) => void;
	onCreate: () => void;
	onDelete: (id: string) => void;
}

export const Sidebar = ({
	sessions,
	activeId,
	onSelect,
	onCreate,
	onDelete,
}: SidebarProps) => {
	return (
		<Flex.Column className="w-52 shrink-0 border-r">
			<Flex.Row align="center" justify="between" className="px-3 py-2">
				<span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
					Sessions
				</span>
				<Button
					size="icon-xs"
					variant="ghost"
					onClick={onCreate}
					aria-label="New session"
				>
					<Plus />
				</Button>
			</Flex.Row>
			<Flex.Column className="flex-1 overflow-y-auto">
				<AnimatePresence initial={false}>
					{sessions.map((session) => (
						<Flex.Row
							key={session.id}
							align="center"
							gap={1}
							appear="slideRight"
							className="group"
							layout
						>
							<Button
								variant={session.id === activeId ? "secondary" : "ghost"}
								className="flex-1 justify-start truncate"
								onClick={() => onSelect(session.id)}
							>
								{session.title}
							</Button>
							{sessions.length > 1 && (
								<Button
									size="icon-xs"
									variant="ghost"
									onClick={() => onDelete(session.id)}
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
	);
};
