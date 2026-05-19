import { kanbanCardsCollection } from "#/lib/kanban-cards-collection";
import { Component } from "../component";
import type { KanbanCard } from "../kanban/model";
import { Button } from "../ui/button";
import { Checkbox } from "../ui/checkbox";
import { Flex } from "../ui/flex";

export const TodoComponent = () => (
	<Component
		name="todos"
		query={(query) => query.from({ card: kanbanCardsCollection })}
	>
		{(cards) => {
			return (cards as KanbanCard[]).map((card) => (
				<Button
					className="h-auto! gap-4 px-4 py-3 text-left"
					variant="outline"
					key={card.id}
				>
					<Flex.Column gap={1} fullWidth>
						<h3>{card.title}</h3>
						<p className="whitespace-break-spaces font-normal text-muted-foreground">
							{card.description}
						</p>
					</Flex.Column>
					<Checkbox
						checked={false}
						onCheckedChange={() => {
							// TODO: Implement checkmark logic
							console.log("checked", card.id);
						}}
					/>
				</Button>
			));
		}}
	</Component>
);
