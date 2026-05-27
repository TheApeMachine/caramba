import { useUser } from "@clerk/tanstack-react-start";
import { ClientOnly } from "@tanstack/react-router";
import { useAll } from "jazz-tools/react";
import { app } from "../../../schema";
import { Loadable, LoadablePending } from "#/components/ui/loadable";
import { parseKanbanAssignees } from "#/lib/parse-kanban-assignees";
import { Button } from "../ui/button";
import { Checkbox } from "../ui/checkbox";
import { Flex } from "../ui/flex";

const TodoList = () => {
	const { user } = useUser();
	const cardRows = useAll(app.kanbanCards.where({}));

	const todoCards = (cardRows ?? []).filter((card) => {
		if (!user || card.column_key === "done") {
			return false;
		}

		return parseKanbanAssignees(card.assignees).includes(user.id);
	});

	return (
		<Loadable
			name="todos"
			isLoading={cardRows === undefined}
			isEmpty={todoCards.length === 0}
			empty={
				<div className="flex h-full items-center justify-center px-3 py-6 text-center text-muted-foreground text-xs">
					No open todos assigned to you.
				</div>
			}
		>
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
		</Loadable>
	);
};

export const TodoComponent = () => (
	<ClientOnly fallback={<LoadablePending name="todos" />}>
		<TodoList />
	</ClientOnly>
);
