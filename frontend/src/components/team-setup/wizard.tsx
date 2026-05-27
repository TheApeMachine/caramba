"use client";

import { useNavigate } from "@tanstack/react-router";
import { ArrowLeftIcon, ArrowRightIcon, CheckIcon } from "lucide-react";
import { useMemo, useState } from "react";
import { type TeamRow, teamCollection } from "#/collections/team";
import { Button } from "#/components/ui/button";
import {
	Card,
	CardFrame,
	CardFrameDescription,
	CardFrameHeader,
	CardFrameTitle,
	CardPanel,
} from "#/components/ui/card";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";
import { cn } from "#/lib/utils";
import { StepDetails } from "./step-details";
import { StepMembers } from "./step-members";
import { StepPrivacy } from "./step-privacy";
import { draftFromTeam, WIZARD_STEPS, type WizardDraft } from "./types";

const StepIndicator = ({ currentIndex }: { currentIndex: number }) => {
	return (
		<Flex.Row className="items-center gap-2">
			{WIZARD_STEPS.map((step, index) => {
				const isActive = index === currentIndex;
				const isComplete = index < currentIndex;

				return (
					<Flex.Row className="items-center gap-2" key={step.id}>
						<Flex.Center
							className={cn(
								"size-7 shrink-0 rounded-full border text-xs font-semibold transition-colors",
								isActive && "border-primary bg-primary text-primary-foreground",
								isComplete && "border-primary/40 bg-primary/10 text-primary",
								!isActive &&
									!isComplete &&
									"border-border text-muted-foreground",
							)}
						>
							{isComplete ? (
								<CheckIcon aria-hidden className="size-3.5" />
							) : (
								index + 1
							)}
						</Flex.Center>
						{index < WIZARD_STEPS.length - 1 ? (
							<div
								className={cn(
									"h-px w-8",
									isComplete ? "bg-primary/40" : "bg-border",
								)}
							/>
						) : null}
					</Flex.Row>
				);
			})}
		</Flex.Row>
	);
};

/*
TeamSetupWizard is the step-driven setup flow new teams land in
immediately after creation. Each step writes its own slice through
teamCollection.update so the user can leave mid-flow and pick back up.
Members are collected client-side here; the actual server-side
membership writes happen on Finish (or are queued for a follow-up
wire-up if the backend endpoint isn't live yet).
*/
export const TeamSetupWizard = ({ team }: { team: TeamRow }) => {
	const navigate = useNavigate();
	const [stepIndex, setStepIndex] = useState(0);
	const [draft, setDraft] = useState<WizardDraft>(() => draftFromTeam(team));
	const [saving, setSaving] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const currentStep = WIZARD_STEPS[stepIndex];
	const isFirst = stepIndex === 0;
	const isLast = stepIndex === WIZARD_STEPS.length - 1;

	const merge = (next: Partial<WizardDraft>) => {
		setDraft((previous) => ({ ...previous, ...next }));
	};

	const persistDraft = async () => {
		const transaction = teamCollection.update(team.id, (existing) => {
			existing.description = draft.description;
			existing.color = draft.color;
			existing.emoji = draft.emoji;
			existing.privacy_mode = draft.privacyMode;
		});

		await transaction.isPersisted.promise;
	};

	const goNext = async () => {
		setError(null);

		try {
			setSaving(true);
			await persistDraft();

			if (isLast) {
				navigate({
					to: "/$orgSlug/$teamSlug",
					params: {
						orgSlug: team.organization_slug,
						teamSlug: team.slug,
					},
				});
				return;
			}

			setStepIndex((index) => index + 1);
		} catch (err) {
			setError(err instanceof Error ? err.message : String(err));
		} finally {
			setSaving(false);
		}
	};

	const goBack = () => {
		setError(null);
		setStepIndex((index) => Math.max(0, index - 1));
	};

	const body = useMemo(() => {
		if (currentStep.id === "details") {
			return <StepDetails draft={draft} onChange={merge} />;
		}

		if (currentStep.id === "members") {
			return <StepMembers draft={draft} onChange={merge} />;
		}

		return <StepPrivacy draft={draft} onChange={merge} />;
	}, [currentStep.id, draft]);

	return (
		<Flex.Column className="mx-auto w-full max-w-2xl gap-6 p-8">
			<Flex.Column gap={3}>
				<Flex.Row className="items-center justify-between">
					<Flex.Column gap={1}>
						<Typography.Span
							className="text-xs font-medium uppercase tracking-wider"
							variant="muted"
						>
							Setting up
						</Typography.Span>
						<Typography.PageTitle className="text-2xl">
							{team.emoji ? `${team.emoji} ${team.name}` : team.name}
						</Typography.PageTitle>
					</Flex.Column>
					<StepIndicator currentIndex={stepIndex} />
				</Flex.Row>
			</Flex.Column>

			<CardFrame>
				<CardFrameHeader>
					<CardFrameTitle>{currentStep.title}</CardFrameTitle>
					<CardFrameDescription>{currentStep.subtitle}</CardFrameDescription>
				</CardFrameHeader>
				<Card>
					<CardPanel>{body}</CardPanel>
				</Card>
			</CardFrame>

			{error ? (
				<Typography.Span variant="error" className="text-sm">
					{error}
				</Typography.Span>
			) : null}

			<Flex.Row className="items-center justify-between">
				<Button
					disabled={isFirst || saving}
					onClick={goBack}
					type="button"
					variant="ghost"
				>
					<ArrowLeftIcon />
					Back
				</Button>
				<Button disabled={saving} onClick={() => void goNext()} type="button">
					{isLast ? (saving ? "Finishing…" : "Finish") : "Continue"}
					{!isLast ? <ArrowRightIcon /> : null}
				</Button>
			</Flex.Row>
		</Flex.Column>
	);
};
