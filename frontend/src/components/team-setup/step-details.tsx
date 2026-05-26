"use client";

import { Field } from "#/components/ui/field";
import { Flex } from "#/components/ui/flex";
import { Input } from "#/components/ui/input";
import { Textarea } from "#/components/ui/textarea";
import { Typography } from "#/components/ui/typography";
import { cn } from "#/lib/utils";
import { TEAM_COLOR_PALETTE, type WizardDraft } from "./types";

/*
StepDetails captures presentation metadata for the team. None of the
fields are required; sensible empty defaults render fine downstream.
*/
export const StepDetails = ({
	draft,
	onChange,
}: {
	draft: WizardDraft;
	onChange: (next: Partial<WizardDraft>) => void;
}) => {
	return (
		<Flex.Column gap={5}>
			<Field>
				<Field.Label htmlFor="team-description">Description</Field.Label>
				<Textarea
					id="team-description"
					onChange={(event) => onChange({ description: event.target.value })}
					placeholder="What is this team focused on?"
					rows={3}
					value={draft.description}
				/>
			</Field>

			<Field>
				<Field.Label htmlFor="team-emoji">Emoji</Field.Label>
				<Input
					id="team-emoji"
					maxLength={4}
					onChange={(event) => onChange({ emoji: event.target.value })}
					placeholder="🧪"
					value={draft.emoji}
				/>
				<Typography.Span className="text-xs" variant="muted">
					Optional. Shown in the switcher and on the dashboard.
				</Typography.Span>
			</Field>

			<Flex.Column gap={2}>
				<Typography.Span className="text-sm font-medium">
					Color
				</Typography.Span>
				<Flex.Row className="flex-wrap gap-2">
					{TEAM_COLOR_PALETTE.map((color) => (
						<button
							aria-label={`Use color ${color}`}
							className={cn(
								"size-7 rounded-full border-2 transition-all",
								draft.color === color
									? "border-foreground scale-110"
									: "border-transparent hover:scale-105",
							)}
							key={color}
							onClick={() =>
								onChange({ color: draft.color === color ? "" : color })
							}
							style={{ backgroundColor: color }}
							type="button"
						/>
					))}
				</Flex.Row>
				<Typography.Span className="text-xs" variant="muted">
					Picks a tag color for projects, cards, and the switcher.
				</Typography.Span>
			</Flex.Column>
		</Flex.Column>
	);
};
