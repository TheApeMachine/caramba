"use client";

import { useUser } from "@clerk/tanstack-react-start";
import { PencilIcon, SaveIcon } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "#/components/ui/avatar";
import { Button } from "#/components/ui/button";
import { Field } from "#/components/ui/field";
import { Flex } from "#/components/ui/flex";
import { Input } from "#/components/ui/input";
import { Textarea } from "#/components/ui/textarea";
import { Typography } from "#/components/ui/typography";
import {
	getResearcherProfile,
	type ResearcherProfileType,
	saveResearcherProfile,
} from "#/server/researcher-profile";

type ProfileDraft = {
	display_name: string;
	role_title: string;
	affiliation: string;
	bio: string;
	website: string;
	research_focus: string;
};

const emptyDraft = (): ProfileDraft => ({
	display_name: "",
	role_title: "",
	affiliation: "",
	bio: "",
	website: "",
	research_focus: "",
});

const clerkDefaults = (
	user: NonNullable<ReturnType<typeof useUser>["user"]>,
): ProfileDraft => ({
	display_name:
		user.fullName ??
		user.username ??
		user.primaryEmailAddress?.emailAddress ??
		"",
	role_title: "",
	affiliation: "",
	bio: "",
	website: "",
	research_focus: "",
});

const mergeProfile = (
	stored: ResearcherProfileType,
	clerkSeed: ProfileDraft,
): ProfileDraft => ({
	display_name: stored.display_name || clerkSeed.display_name,
	role_title: stored.role_title,
	affiliation: stored.affiliation,
	bio: stored.bio,
	website: stored.website,
	research_focus: stored.research_focus,
});

export const ResearcherProfileWidget = () => {
	const { user, isLoaded } = useUser();
	const [editing, setEditing] = useState(false);
	const [draft, setDraft] = useState<ProfileDraft>(emptyDraft);
	const [loading, setLoading] = useState(true);
	const [saving, setSaving] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const loadProfile = useCallback(async () => {
		if (!user) {
			return;
		}

		setLoading(true);
		setError(null);

		try {
			const stored = await getResearcherProfile();
			const seed = clerkDefaults(user);
			setDraft(mergeProfile(stored, seed));
		} catch (loadError) {
			setDraft(clerkDefaults(user));
			setError(
				loadError instanceof Error ? loadError.message : String(loadError),
			);
		} finally {
			setLoading(false);
		}
	}, [user]);

	useEffect(() => {
		if (!isLoaded || !user) {
			return;
		}

		void loadProfile();
	}, [isLoaded, user, loadProfile]);

	const handleSave = async () => {
		setSaving(true);
		setError(null);

		try {
			await saveResearcherProfile({ data: draft });
			setEditing(false);
			await loadProfile();
		} catch (saveError) {
			setError(
				saveError instanceof Error ? saveError.message : String(saveError),
			);
		} finally {
			setSaving(false);
		}
	};

	if (!isLoaded || loading) {
		return (
			<Flex.Center className="h-full p-4">
				<Typography.Paragraph variant="muted">
					Loading profile…
				</Typography.Paragraph>
			</Flex.Center>
		);
	}

	if (!user) {
		return (
			<Flex.Center className="h-full p-4">
				<Typography.Paragraph variant="muted">
					Sign in to manage your researcher profile.
				</Typography.Paragraph>
			</Flex.Center>
		);
	}

	const email = user.primaryEmailAddress?.emailAddress ?? "";
	const displayLabel = draft.display_name.trim() || "Researcher";
	const initials = displayLabel.slice(0, 2).toUpperCase();

	return (
		<Flex.Column gap={3} className="h-full p-3">
			<Flex.Row className="items-start gap-3">
				<Avatar className="size-14 border-2 border-primary/20">
					{user.imageUrl ? <AvatarImage alt="" src={user.imageUrl} /> : null}
					<AvatarFallback className="text-base">{initials}</AvatarFallback>
				</Avatar>
				<Flex.Column gap={1} className="min-w-0 flex-1">
					<Typography.H4 variant="sectionHeading">
						Researcher profile
					</Typography.H4>
					<Typography.Paragraph variant="muted">
						How you show up on cards, papers, and collaboration — separate from
						your account menu.
					</Typography.Paragraph>
					{email ? (
						<Typography.Small variant="muted" className="truncate">
							Account: {email}
						</Typography.Small>
					) : null}
				</Flex.Column>
				<Button
					type="button"
					variant={editing ? "outline" : "ghost"}
					size="icon-sm"
					aria-label={editing ? "Cancel editing profile" : "Edit profile"}
					onClick={() => {
						if (editing) {
							void loadProfile();
						}

						setEditing((current) => !current);
					}}
				>
					<PencilIcon className="size-4" />
				</Button>
			</Flex.Row>

			{error ? (
				<Typography.Paragraph className="text-destructive text-xs">
					{error}
				</Typography.Paragraph>
			) : null}

			{editing ? (
				<Flex.Column gap={3} className="min-h-0 flex-1 overflow-y-auto">
					<Field>
						<Field.Label htmlFor="profile-display-name">
							Display name
						</Field.Label>
						<Input
							id="profile-display-name"
							value={draft.display_name}
							onChange={(event) =>
								setDraft((current) => ({
									...current,
									display_name: event.target.value,
								}))
							}
						/>
					</Field>
					<Field>
						<Field.Label htmlFor="profile-role">Role / title</Field.Label>
						<Input
							id="profile-role"
							value={draft.role_title}
							onChange={(event) =>
								setDraft((current) => ({
									...current,
									role_title: event.target.value,
								}))
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
								setDraft((current) => ({
									...current,
									affiliation: event.target.value,
								}))
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
								setDraft((current) => ({
									...current,
									research_focus: event.target.value,
								}))
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
								setDraft((current) => ({
									...current,
									website: event.target.value,
								}))
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
								setDraft((current) => ({
									...current,
									bio: event.target.value,
								}))
							}
							placeholder="A sentence or two collaborators will see."
						/>
					</Field>
					<Button
						type="button"
						disabled={saving}
						onClick={() => {
							void handleSave();
						}}
					>
						<SaveIcon className="size-4" />
						{saving ? "Saving…" : "Save profile"}
					</Button>
				</Flex.Column>
			) : (
				<Flex.Column gap={2} className="min-h-0 flex-1 text-sm">
					<div>
						<div className="font-semibold text-foreground text-base">
							{displayLabel}
						</div>
						{draft.role_title ? (
							<div className="text-muted-foreground">{draft.role_title}</div>
						) : null}
						{draft.affiliation ? (
							<div className="text-muted-foreground text-xs">
								{draft.affiliation}
							</div>
						) : null}
					</div>
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
							Add your role, affiliation, and focus so teammates know who you
							are beyond the header avatar.
						</Typography.Paragraph>
					) : null}
				</Flex.Column>
			)}
		</Flex.Column>
	);
};
