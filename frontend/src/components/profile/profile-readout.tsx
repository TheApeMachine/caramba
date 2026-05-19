"use client";

import type { ProfileDraft } from "#/components/profile/model";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

interface ProfileReadoutProps {
	draft: ProfileDraft;
}

export const ProfileReadout = ({ draft }: ProfileReadoutProps) => (
	<Flex.Column gap={2} className="min-h-0 flex-1 text-sm">
		{draft.research_focus ? (
			<div>
				<div className="text-muted-foreground text-xs uppercase tracking-wide">
					Focus
				</div>
				<div>{draft.research_focus}</div>
			</div>
		) : null}
		{draft.bio ? (
			<div>
				<div className="text-muted-foreground text-xs uppercase tracking-wide">
					Bio
				</div>
				<p className="text-muted-foreground leading-snug">{draft.bio}</p>
			</div>
		) : null}
		{draft.website ? (
			<a
				className="truncate text-primary text-xs underline-offset-4 hover:underline"
				href={draft.website}
				rel="noreferrer"
				target="_blank"
			>
				{draft.website}
			</a>
		) : null}
		{!draft.role_title &&
		!draft.affiliation &&
		!draft.bio &&
		!draft.research_focus ? (
			<Typography.Paragraph variant="muted">
				Add your role, affiliation, and focus so teammates know who you are
				beyond the header avatar.
			</Typography.Paragraph>
		) : null}
	</Flex.Column>
);
