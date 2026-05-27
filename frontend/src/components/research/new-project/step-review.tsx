"use client";

import type { NewResearchProjectSpec } from "#/components/research/new-project/model";
import { Typography } from "#/components/ui/typography";

/*
StepReview is the final confirmation copy. The actual review surface
lives in the preview pane rendered by the wizard host.
*/
export const StepReview = ({
	draft,
}: {
	draft: NewResearchProjectSpec;
}) => {
	if (draft.papers.length === 0) {
		return (
			<Typography.Paragraph variant="muted">
				Launch creates the research project, team memberships, starter Kanban
				cards, and no papers yet (add them later from the editor).
			</Typography.Paragraph>
		);
	}

	const paperWord = draft.papers.length === 1 ? "paper" : "papers";

	return (
		<Typography.Paragraph variant="muted">
			Launch creates the research project, team memberships, starter Kanban
			cards, and {draft.papers.length} linked {paperWord}.
		</Typography.Paragraph>
	);
};
