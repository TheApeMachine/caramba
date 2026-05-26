"use client";

import { AlertTriangleIcon, CloudIcon, LockIcon } from "lucide-react";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";
import { cn } from "#/lib/utils";
import type { WizardDraft } from "./types";

const OPTIONS = [
	{
		value: "shared" as const,
		title: "Shared",
		subtitle: "Cloud-synced, collaborative",
		icon: CloudIcon,
		bullets: [
			"Team data syncs across every member's device in real time.",
			"Stored in this server's database.",
			"Recommended for any team with more than one person.",
		],
	},
	{
		value: "local" as const,
		title: "Local-only",
		subtitle: "Private to this device",
		icon: LockIcon,
		bullets: [
			"Nothing about this team — projects, cards, members — ever leaves your browser.",
			"You're the only one who can see it.",
			"If you clear your browser data, the team is gone.",
		],
	},
];

/*
StepPrivacy is the most architecturally significant choice in the
wizard: shared teams sync via Electric and live in Postgres, local
teams stay entirely in the creator's browser via localStorage-backed
collections.
*/
export const StepPrivacy = ({
	draft,
	onChange,
}: {
	draft: WizardDraft;
	onChange: (next: Partial<WizardDraft>) => void;
}) => {
	return (
		<Flex.Column gap={4}>
			<Flex.Column gap={3}>
				{OPTIONS.map((option) => {
					const Icon = option.icon;
					const isSelected = draft.privacyMode === option.value;

					return (
						<button
							className={cn(
								"flex flex-col gap-3 rounded-2xl border p-4 text-left transition-all",
								isSelected
									? "border-primary bg-primary/5 shadow-sm"
									: "border-border hover:border-ring/40 hover:bg-muted/40",
							)}
							key={option.value}
							onClick={() => onChange({ privacyMode: option.value })}
							type="button"
						>
							<Flex.Row className="items-center gap-3">
								<Flex.Center
									className={cn(
										"size-10 shrink-0 rounded-xl border",
										isSelected
											? "border-primary/40 bg-primary/10 text-primary"
											: "border-border bg-muted/40 text-muted-foreground",
									)}
								>
									<Icon aria-hidden className="size-5" />
								</Flex.Center>
								<Flex.Column className="min-w-0 flex-1 gap-0.5">
									<Typography.Span className="font-semibold text-sm">
										{option.title}
									</Typography.Span>
									<Typography.Span className="text-xs" variant="muted">
										{option.subtitle}
									</Typography.Span>
								</Flex.Column>
							</Flex.Row>
							<ul className="ml-1 list-disc space-y-1 pl-5 text-muted-foreground text-xs">
								{option.bullets.map((bullet) => (
									<li key={bullet}>{bullet}</li>
								))}
							</ul>
						</button>
					);
				})}
			</Flex.Column>

			{draft.privacyMode === "local" ? (
				<Flex.Row className="items-start gap-2 rounded-lg border border-warning/40 bg-warning/5 p-3">
					<AlertTriangleIcon
						aria-hidden
						className="mt-0.5 size-4 shrink-0 text-warning"
					/>
					<Typography.Paragraph className="text-xs" variant="muted">
						Local teams cannot be shared, exported through Electric, or recovered
						if browser storage is cleared. Switching to Shared later requires
						re-creating the team.
					</Typography.Paragraph>
				</Flex.Row>
			) : null}
		</Flex.Column>
	);
};
