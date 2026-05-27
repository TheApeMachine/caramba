"use client";

import { useOrganization } from "@clerk/tanstack-react-start";
import { UsersIcon } from "lucide-react";
import { useMemo } from "react";
import { SelectionCard } from "#/components/benchmarks/selection-card";
import type { NewResearchProjectSpec } from "#/components/research/new-project/model";
import { Checkbox } from "#/components/ui/checkbox";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

type OrgMember = { userId: string; displayName: string };

const collectOrgMembers = (
	rows: ReadonlyArray<{
		publicUserData?: {
			userId?: string;
			firstName?: string | null;
			lastName?: string | null;
			identifier?: string;
		};
	}>,
): OrgMember[] =>
	rows
		.map((membership) => {
			const userId = membership.publicUserData?.userId;

			if (!userId) {
				return null;
			}

			const displayName =
				[
					membership.publicUserData?.firstName,
					membership.publicUserData?.lastName,
				]
					.filter(Boolean)
					.join(" ")
					.trim() ||
				membership.publicUserData?.identifier ||
				userId;

			return { userId, displayName };
		})
		.filter((entry): entry is OrgMember => Boolean(entry));

/*
StepTeam shows organization members as selection cards. The current
user is always present in draft.memberIds (set by the wizard caller
before mount) and renders as a disabled, always-on owner row.
*/
export const StepTeam = ({
	draft,
	merge,
	currentUserId,
	currentUserLabel,
}: {
	draft: NewResearchProjectSpec;
	merge: (patch: Partial<NewResearchProjectSpec>) => void;
	currentUserId: string;
	currentUserLabel: string;
}) => {
	const { organization, memberships } = useOrganization({
		memberships: { pageSize: 50 },
	});

	const orgMembers = useMemo(
		() => collectOrgMembers(memberships?.data ?? []),
		[memberships?.data],
	);

	const toggleMember = (memberId: string) => {
		if (memberId === currentUserId) {
			return;
		}

		const isSelected = draft.memberIds.includes(memberId);

		merge({
			memberIds: isSelected
				? draft.memberIds.filter((entry) => entry !== memberId)
				: [...draft.memberIds, memberId],
		});
	};

	if (orgMembers.length === 0) {
		return (
			<Flex.Column gap={2}>
				<Typography.Paragraph variant="muted">
					{organization
						? `${organization.name} has no other members yet. You will be the project owner.`
						: "Personal workspace — you will be the project owner."}
				</Typography.Paragraph>
				{currentUserId ? (
					<Flex.Row
						align="center"
						gap={2}
						className="rounded-lg border bg-background/60 px-3 py-2"
					>
						<Checkbox checked disabled />
						<Typography.Span className="text-sm">
							{currentUserLabel} (owner)
						</Typography.Span>
					</Flex.Row>
				) : null}
			</Flex.Column>
		);
	}

	return (
		<div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
			{orgMembers.map((member) => {
				const selected = draft.memberIds.includes(member.userId);
				const isOwner = member.userId === currentUserId;

				return (
					<SelectionCard
						key={member.userId}
						selected={selected}
						disabled={isOwner}
						onSelect={() => toggleMember(member.userId)}
						title={member.displayName}
						subtitle={isOwner ? "Project owner" : "Collaborator"}
						icon={<UsersIcon className="size-4" />}
						hint={
							isOwner
								? "Always on the project"
								: selected
									? "Included on the board"
									: "Tap to add"
						}
					/>
				);
			})}
		</div>
	);
};
