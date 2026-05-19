"use client";

import { useUser } from "@clerk/tanstack-react-start";
import { useCallback, useEffect, useState } from "react";
import {
	ProfileComponent,
	type ProfileDraft,
	type ProfileExpansion,
} from "#/components/profile/component";
import {
	clerkProfileDefaults,
	emptyProfileDraft,
	isUnsetResearcherProfile,
	mergeStoredProfile,
} from "#/components/profile/model";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";
import {
	getResearcherProfile,
	saveResearcherProfile,
} from "#/server/researcher-profile";

export const ResearcherProfileWidget = () => {
	const { user, isLoaded } = useUser();
	const [expansion, setExpansion] = useState<ProfileExpansion>("summary");
	const [draft, setDraft] = useState<ProfileDraft>(emptyProfileDraft);
	const [loading, setLoading] = useState(true);
	const [saving, setSaving] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const loadProfile = useCallback(async () => {
		if (!user) {
			return;
		}

		setLoading(true);
		setError(null);

		const seed = clerkProfileDefaults(user);

		try {
			let stored = await getResearcherProfile();

			if (isUnsetResearcherProfile(stored)) {
				const bootstrapDraft = mergeStoredProfile(stored, seed);
				await saveResearcherProfile({ data: bootstrapDraft });
				stored = await getResearcherProfile();
			}

			setDraft(mergeStoredProfile(stored, seed));
		} catch (loadError) {
			setDraft(seed);
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
			setExpansion("summary");
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

	return (
		<ProfileComponent
			user={user}
			expansion={expansion}
			onExpansionChange={setExpansion}
			draft={draft}
			onDraftChange={setDraft}
			description="How you show up on cards, papers, and collaboration — separate from your account menu."
			error={error}
			saving={saving}
			showEditControl
			onSave={() => {
				void handleSave();
			}}
			onCancelEdit={() => {
				void loadProfile();
			}}
			className="h-full w-full border-0 bg-transparent shadow-none backdrop-blur-none"
		/>
	);
};
