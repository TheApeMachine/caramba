import { useUser } from "@clerk/tanstack-react-start";
import { kanbanCardsCollection } from "#/lib/kanban-cards-collection";
import { parseKanbanAssignees } from "#/lib/parse-kanban-assignees";
import { Component } from "../component";
import type { KanbanCard } from "../kanban/model";
import { Button } from "../ui/button";
import { Checkbox } from "../ui/checkbox";
import { Flex } from "../ui/flex";

export const TodoComponent = () => {
	const { user } = useUser();

	return (
		<Component
			name="todos"
			query={(query) => query.from({ card: kanbanCardsCollection })}
		>
			{(cards) => {
				const todoCards = (cards as KanbanCard[]).filter((card) => {
					if (!user || card.column_key === "done") {
						return false;
					}

					return parseKanbanAssignees(card.assignees_json).includes(user.id);
				});

				if (todoCards.length === 0) {
					return (
						<div className="flex h-full items-center justify-center px-3 py-6 text-center text-muted-foreground text-xs">
							No open todos assigned to you.
						</div>
					);
				}

				return (
					<Flex.Column gap={2}>
						{todoCards.map((card) => (
							<Button
								className="h-auto! gap-4 px-4 py-3 text-left"
								variant="outline"
								key={card.id}
							>
								<Flex.Column gap={1} fullWidth>
									<h3>{card.title}</h3>
									{card.description ? (
										<p className="whitespace-break-spaces font-normal text-muted-foreground">
											{card.description}
										</p>
									) : null}
								</Flex.Column>
								<Checkbox
									checked={false}
									onCheckedChange={() => {
										// TODO: Implement checkmark logic
										console.log("checked", card.id);
									}}
								/>
							</Button>
						))}
					</Flex.Column>
				);
			}}
		</Component>
	);
};
