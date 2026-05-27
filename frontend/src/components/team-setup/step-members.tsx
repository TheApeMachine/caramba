"use client";

import { useOrganization } from "@clerk/tanstack-react-start";
import { CheckIcon } from "lucide-react";
import { Avatar, AvatarFallback } from "#/components/ui/avatar";
import { Empty } from "#/components/ui/empty";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";
import { cn } from "#/lib/utils";
import type { WizardDraft } from "./types";

/*
StepMembers lists the current Clerk organization's members so the
creator can pick teammates. Selections are kept in wizard state; the
actual server-side membership writes happen on finish (handled by the
wizard controller).
*/
export const StepMembers = ({
	draft,
	onChange,
}: {
	draft: WizardDraft;
	onChange: (next: Partial<WizardDraft>) => void;
}) => {
	const { memberships, isLoaded } = useOrganization({
		memberships: { infinite: true, pageSize: 50 },
	});

	const rows = memberships?.data ?? [];
	const selected = new Set(draft.memberIds);

	const toggle = (userId: string) => {
		const next = new Set(selected);

		if (next.has(userId)) {
			next.delete(userId);
		} else {
			next.add(userId);
		}

		onChange({ memberIds: Array.from(next) });
	};

	if (!isLoaded) {
		return (
			<Flex.Center className="py-12">
				<Typography.Paragraph variant="muted">
					Loading organization members…
				</Typography.Paragraph>
			</Flex.Center>
		);
	}

	if (rows.length === 0) {
		return (
			<Empty>
				<Empty.Header>
					<Empty.Title>No teammates yet</Empty.Title>
					<Empty.Description>
						You're the only member of this organization. You can invite people
						to the org first, then come back to add them to the team.
					</Empty.Description>
				</Empty.Header>
			</Empty>
		);
	}

	return (
		<Flex.Column gap={3}>
			<Typography.Paragraph className="text-sm" variant="muted">
				Pick teammates to add as members. You'll always be added as the owner.
			</Typography.Paragraph>
			<Flex.Column gap={1}>
				{rows.map((membership) => {
					const userId = membership.publicUserData?.userId ?? "";
					const name =
						membership.publicUserData?.firstName ||
						membership.publicUserData?.identifier ||
						"Unknown";
					const initials = name
						.split(/\s+/)
						.slice(0, 2)
						.map((part) => part[0]?.toUpperCase() ?? "")
						.join("");
					const isSelected = selected.has(userId);

					return (
						<button
							className={cn(
								"flex items-center gap-3 rounded-lg border px-3 py-2 text-left transition-colors",
								isSelected
									? "border-primary/40 bg-primary/10"
									: "border-border hover:bg-muted/50",
							)}
							key={membership.id}
							onClick={() => toggle(userId)}
							type="button"
						>
							<Avatar className="size-7 text-xs">
								<AvatarFallback>{initials || "?"}</AvatarFallback>
							</Avatar>
							<Flex.Column className="min-w-0 flex-1 gap-0.5">
								<Typography.Span className="truncate text-sm">
									{name}
								</Typography.Span>
								{membership.publicUserData?.identifier ? (
									<Typography.Span className="truncate text-xs" variant="muted">
										{membership.publicUserData.identifier}
									</Typography.Span>
								) : null}
							</Flex.Column>
							{isSelected ? (
								<CheckIcon aria-hidden className="size-4 text-primary" />
							) : null}
						</button>
					);
				})}
			</Flex.Column>
		</Flex.Column>
	);
};
