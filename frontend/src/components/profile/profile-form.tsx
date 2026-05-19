"use client";

import { SaveIcon } from "lucide-react";
import type { ProfileDraft } from "#/components/profile/model";
import { Button } from "#/components/ui/button";
import { Field } from "#/components/ui/field";
import { Flex } from "#/components/ui/flex";
import { Input } from "#/components/ui/input";
import { Textarea } from "#/components/ui/textarea";

interface ProfileFormProps {
	draft: ProfileDraft;
	saving: boolean;
	onDraftChange: (draft: ProfileDraft) => void;
	onSave: () => void;
}

export const ProfileForm = ({
	draft,
	saving,
	onDraftChange,
	onSave,
}: ProfileFormProps) => (
	<Flex.Column gap={3} className="min-h-0 flex-1 overflow-y-auto">
		<Field>
			<Field.Label htmlFor="profile-display-name">Display name</Field.Label>
			<Input
				id="profile-display-name"
				value={draft.display_name}
				onChange={(event) =>
					onDraftChange({ ...draft, display_name: event.target.value })
				}
			/>
		</Field>
		<Field>
			<Field.Label htmlFor="profile-role">Role / title</Field.Label>
			<Input
				id="profile-role"
				value={draft.role_title}
				onChange={(event) =>
					onDraftChange({ ...draft, role_title: event.target.value })
				}
				placeholder="e.g. PhD researcher, Staff ML engineer"
			/>
		</Field>
		<Field>
			<Field.Label htmlFor="profile-affiliation">Affiliation</Field.Label>
			<Input
				id="profile-affiliation"
				value={draft.affiliation}
				onChange={(event) =>
					onDraftChange({ ...draft, affiliation: event.target.value })
				}
				placeholder="Lab, university, or org"
			/>
		</Field>
		<Field>
			<Field.Label htmlFor="profile-focus">Research focus</Field.Label>
			<Input
				id="profile-focus"
				value={draft.research_focus}
				onChange={(event) =>
					onDraftChange({ ...draft, research_focus: event.target.value })
				}
				placeholder="e.g. sparse attention, vision-language"
			/>
		</Field>
		<Field>
			<Field.Label htmlFor="profile-website">Website</Field.Label>
			<Input
				id="profile-website"
				value={draft.website}
				onChange={(event) =>
					onDraftChange({ ...draft, website: event.target.value })
				}
				placeholder="https://"
			/>
		</Field>
		<Field>
			<Field.Label htmlFor="profile-bio">Bio</Field.Label>
			<Textarea
				id="profile-bio"
				rows={3}
				value={draft.bio}
				onChange={(event) =>
					onDraftChange({ ...draft, bio: event.target.value })
				}
				placeholder="A sentence or two collaborators will see."
			/>
		</Field>
		<Button type="button" disabled={saving} onClick={onSave}>
			<SaveIcon className="size-4" />
			{saving ? "Saving…" : "Save profile"}
		</Button>
	</Flex.Column>
);
