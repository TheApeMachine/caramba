"use client";

import { Field } from "#/components/ui/field";
import { Flex } from "#/components/ui/flex";
import { Input } from "#/components/ui/input";
import { Textarea } from "#/components/ui/textarea";
import { deriveProjectSlug } from "#/components/research/project-slug";
import type { NewResearchProjectSpec } from "#/components/research/new-project/model";

/*
StepBasics captures the project name, free-text description, and the
URL slug. Slug auto-derives from the name placeholder but the user can
override it directly.
*/
export const StepBasics = ({
	draft,
	merge,
}: {
	draft: NewResearchProjectSpec;
	merge: (patch: Partial<NewResearchProjectSpec>) => void;
}) => (
	<Flex.Column gap={3}>
		<Field>
			<Field.Label htmlFor="project-name">Name</Field.Label>
			<Input
				id="project-name"
				value={draft.name}
				onChange={(event) => merge({ name: event.target.value })}
				placeholder="e.g. Sparse attention ablations"
			/>
		</Field>
		<Field>
			<Field.Label htmlFor="project-description">Description</Field.Label>
			<Textarea
				id="project-description"
				value={draft.description}
				onChange={(event) => merge({ description: event.target.value })}
				placeholder="What are you trying to learn or ship?"
				rows={4}
			/>
		</Field>
		<Field>
			<Field.Label htmlFor="project-slug">Board slug</Field.Label>
			<Input
				id="project-slug"
				value={draft.projectSlug}
				onChange={(event) => merge({ projectSlug: event.target.value })}
				placeholder={deriveProjectSlug(draft.name) || "my-project"}
			/>
			<Field.Description>
				Used in URLs as /{deriveProjectSlug(draft.projectSlug || draft.name)}
			</Field.Description>
		</Field>
	</Flex.Column>
);
