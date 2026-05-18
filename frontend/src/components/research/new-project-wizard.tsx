"use client";

import { useOrganization, useUser } from "@clerk/tanstack-react-start";
import { useNavigate } from "@tanstack/react-router";
import {
	CheckIcon,
	FileTextIcon,
	FolderKanban,
	PlayIcon,
	UsersIcon,
	XIcon,
} from "lucide-react";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import { SelectionCard } from "#/components/benchmarks/selection-card";
import {
	emptyNewResearchProjectSpec,
	type NewResearchProjectSpec,
} from "#/components/research/new-project-model";
import { NewProjectPapersStep } from "#/components/research/new-project-papers-step";
import { NewProjectPreview } from "#/components/research/new-project-preview";
import { deriveProjectSlug } from "#/components/research/project-slug";
import { Alert, AlertDescription, AlertTitle } from "#/components/ui/alert";
import { Badge } from "#/components/ui/badge";
import { Button } from "#/components/ui/button";
import { Checkbox } from "#/components/ui/checkbox";
import { Field } from "#/components/ui/field";
import { Flex } from "#/components/ui/flex";
import { Input } from "#/components/ui/input";
import { Textarea } from "#/components/ui/textarea";
import { Typography } from "#/components/ui/typography";
import { provisionResearchProject } from "#/server/provision-research-project";
import { cn } from "@/lib/utils";

const SECTION_IDS = ["basics", "team", "papers", "review"] as const;
type SectionId = (typeof SECTION_IDS)[number];

const sectionLabel: Record<SectionId, string> = {
	basics: "Basics",
	team: "Team",
	papers: "Papers",
	review: "Review",
};

const sectionIcon: Record<SectionId, ReactNode> = {
	basics: <FolderKanban className="size-4" />,
	team: <UsersIcon className="size-4" />,
	papers: <FileTextIcon className="size-4" />,
	review: <CheckIcon className="size-4" />,
};

const isComplete = (
	spec: NewResearchProjectSpec,
	section: SectionId,
): boolean => {
	if (section === "basics") {
		return Boolean(spec.name.trim());
	}

	if (section === "team") {
		return spec.memberIds.length > 0;
	}

	if (section === "papers") {
		return true;
	}

	return Boolean(spec.name.trim() && spec.memberIds.length > 0);
};

const readyToLaunch = (spec: NewResearchProjectSpec): boolean =>
	SECTION_IDS.every((section) => isComplete(spec, section));

const SectionShell = ({
	sectionId,
	title,
	hint,
	complete,
	children,
}: {
	sectionId: SectionId;
	title: string;
	hint: string;
	complete: boolean;
	children: ReactNode;
}) => (
	<section
		id={`section-${sectionId}`}
		className="flex scroll-mt-4 flex-col gap-3 rounded-2xl border bg-card/40 p-4"
	>
		<Flex.Row className="items-start gap-2">
			<span
				className={cn(
					"flex size-7 shrink-0 items-center justify-center rounded-full border transition",
					complete
						? "border-primary bg-primary text-primary-foreground"
						: "border-muted-foreground/30 bg-background text-muted-foreground",
				)}
			>
				{complete ? <CheckIcon className="size-4" /> : sectionIcon[sectionId]}
			</span>
			<Flex.Column gap={1}>
				<h2 className="font-semibold text-foreground text-sm">{title}</h2>
				<Typography.Paragraph variant="muted">{hint}</Typography.Paragraph>
			</Flex.Column>
		</Flex.Row>
		{children}
	</section>
);

const Stepper = ({
	spec,
	onJump,
}: {
	spec: NewResearchProjectSpec;
	onJump: (section: SectionId) => void;
}) => (
	<ol className="flex flex-wrap items-center gap-1">
		{SECTION_IDS.map((sectionId, index) => {
			const complete = isComplete(spec, sectionId);

			return (
				<li key={sectionId} className="flex items-center gap-1">
					<button
						type="button"
						onClick={() => onJump(sectionId)}
						className={cn(
							"flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs transition",
							complete
								? "border-primary/40 bg-primary/5 text-foreground"
								: "border-transparent text-muted-foreground hover:bg-muted/40",
						)}
					>
						<span
							className={cn(
								"flex size-5 items-center justify-center rounded-full font-medium text-[10px]",
								complete
									? "bg-primary text-primary-foreground"
									: "bg-muted text-muted-foreground",
							)}
						>
							{complete ? <CheckIcon className="size-3" /> : index + 1}
						</span>
						{sectionLabel[sectionId]}
					</button>
					{index < SECTION_IDS.length - 1 ? (
						<span
							aria-hidden
							className="h-px w-3 bg-muted-foreground/30 sm:w-4"
						/>
					) : null}
				</li>
			);
		})}
	</ol>
);

export const NewProjectWizard = () => {
	const navigate = useNavigate();
	const { user, isLoaded: userLoaded } = useUser();
	const {
		organization,
		memberships,
		isLoaded: organizationLoaded,
	} = useOrganization({
		memberships: {
			pageSize: 50,
		},
	});

	const [spec, setSpec] = useState<NewResearchProjectSpec>(() =>
		emptyNewResearchProjectSpec(),
	);
	const [persistError, setPersistError] = useState<string | null>(null);
	const [launching, setLaunching] = useState(false);

	const currentUserId = user?.id ?? "";

	const orgMembers = useMemo(() => {
		const rows = memberships?.data ?? [];

		return rows
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
			.filter((entry): entry is { userId: string; displayName: string } =>
				Boolean(entry),
			);
	}, [memberships?.data]);

	useEffect(() => {
		if (!userLoaded || !currentUserId) {
			return;
		}

		setSpec((current) => {
			if (current.memberIds.includes(currentUserId)) {
				return current;
			}

			return {
				...current,
				memberIds: [currentUserId, ...current.memberIds],
			};
		});
	}, [currentUserId, userLoaded]);

	const memberLabels = useMemo(() => {
		const labels = new Map<string, string>();

		if (currentUserId) {
			const selfLabel =
				[user?.firstName, user?.lastName].filter(Boolean).join(" ").trim() ||
				user?.primaryEmailAddress?.emailAddress ||
				"You";

			labels.set(currentUserId, selfLabel);
		}

		for (const member of orgMembers) {
			labels.set(member.userId, member.displayName);
		}

		return labels;
	}, [currentUserId, orgMembers, user]);

	const toggleMember = (memberId: string) => {
		if (memberId === currentUserId) {
			return;
		}

		setSpec((current) => {
			const selected = current.memberIds.includes(memberId);

			return {
				...current,
				memberIds: selected
					? current.memberIds.filter((entry) => entry !== memberId)
					: [...current.memberIds, memberId],
			};
		});
	};

	const scrollTo = (section: SectionId) => {
		document
			.getElementById(`section-${section}`)
			?.scrollIntoView({ behavior: "smooth", block: "start" });
	};

	const launch = async () => {
		if (!readyToLaunch(spec) || launching) {
			return;
		}

		setPersistError(null);
		setLaunching(true);

		try {
			const slug = deriveProjectSlug(spec.projectSlug || spec.name);

			await provisionResearchProject({
				data: {
					id: spec.id,
					name: spec.name.trim(),
					description: spec.description.trim(),
					project_slug: slug,
					member_ids: spec.memberIds,
					papers: spec.papers.map((paper) => ({
						id: paper.id,
						title: paper.title.trim(),
					})),
				},
			});

			await navigate({
				to: "/kanban/project/$projectId",
				params: { projectId: spec.id },
			});
		} catch (error) {
			setPersistError(error instanceof Error ? error.message : String(error));
		} finally {
			setLaunching(false);
		}
	};

	const completedCount = useMemo(
		() => SECTION_IDS.filter((section) => isComplete(spec, section)).length,
		[spec],
	);

	const teamReady = userLoaded && organizationLoaded;

	return (
		<Flex.Column gap={4} className="h-full min-h-0 flex-1">
			<header className="flex flex-wrap items-center justify-between gap-3">
				<Flex.Column gap={1}>
					<Typography.PageTitle>New research project</Typography.PageTitle>
					<Typography.Paragraph variant="muted">
						Name the effort, invite collaborators, link one or more papers, and
						spin up a Kanban board with starter cards in one step.
					</Typography.Paragraph>
				</Flex.Column>
				<Flex.Row className="items-center gap-2">
					<Badge variant="outline" size="lg">
						{completedCount}/{SECTION_IDS.length} steps complete
					</Badge>
					<Button
						type="button"
						variant="outline"
						onClick={() => navigate({ to: "/research" })}
					>
						<XIcon /> Cancel
					</Button>
					<Button
						type="button"
						onClick={launch}
						disabled={!readyToLaunch(spec) || launching || !teamReady}
					>
						<PlayIcon /> {launching ? "Creating…" : "Create project"}
					</Button>
				</Flex.Row>
			</header>

			{persistError ? (
				<Alert variant="error">
					<AlertTitle>Could not create project</AlertTitle>
					<AlertDescription>{persistError}</AlertDescription>
				</Alert>
			) : null}

			<Stepper spec={spec} onJump={scrollTo} />

			<div className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(320px,400px)]">
				<Flex.Column gap={4} className="min-h-0 overflow-y-auto pr-1">
					<SectionShell
						sectionId="basics"
						title="Project basics"
						hint="Name and describe the research effort. The slug routes your Kanban board."
						complete={isComplete(spec, "basics")}
					>
						<Field>
							<Field.Label htmlFor="project-name">Name</Field.Label>
							<Input
								id="project-name"
								value={spec.name}
								onChange={(event) =>
									setSpec((current) => ({
										...current,
										name: event.target.value,
									}))
								}
								placeholder="e.g. Sparse attention ablations"
							/>
						</Field>
						<Field>
							<Field.Label htmlFor="project-description">
								Description
							</Field.Label>
							<Textarea
								id="project-description"
								value={spec.description}
								onChange={(event) =>
									setSpec((current) => ({
										...current,
										description: event.target.value,
									}))
								}
								placeholder="What are you trying to learn or ship?"
								rows={4}
							/>
						</Field>
						<Field>
							<Field.Label htmlFor="project-slug">Board slug</Field.Label>
							<Input
								id="project-slug"
								value={spec.projectSlug}
								onChange={(event) =>
									setSpec((current) => ({
										...current,
										projectSlug: event.target.value,
									}))
								}
								placeholder={deriveProjectSlug(spec.name) || "my-project"}
							/>
							<Field.Description>
								Used in URLs as /
								{deriveProjectSlug(spec.projectSlug || spec.name)}
							</Field.Description>
						</Field>
					</SectionShell>

					<SectionShell
						sectionId="team"
						title="Team members"
						hint={
							organization
								? `Assign collaborators from ${organization.name}. You are always included as the project owner.`
								: "Personal workspace — you will be the project owner."
						}
						complete={isComplete(spec, "team")}
					>
						{!teamReady ? (
							<Typography.Paragraph variant="muted">
								Loading team roster…
							</Typography.Paragraph>
						) : null}

						{teamReady && orgMembers.length === 0 ? (
							<div className="rounded-xl border border-dashed bg-background/50 p-3">
								<Typography.Paragraph variant="muted">
									Only you are on this project. Switch to an organization in the
									header menu to invite teammates.
								</Typography.Paragraph>
							</div>
						) : null}

						{teamReady && orgMembers.length > 0 ? (
							<div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
								{orgMembers.map((member) => {
									const selected = spec.memberIds.includes(member.userId);
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
						) : null}

						{teamReady && orgMembers.length === 0 && currentUserId ? (
							<Flex.Row className="items-center gap-2 rounded-lg border bg-background/60 px-3 py-2">
								<Checkbox checked disabled />
								<span className="text-sm">
									{memberLabels.get(currentUserId) ?? "You"} (owner)
								</span>
							</Flex.Row>
						) : null}
					</SectionShell>

					<SectionShell
						sectionId="papers"
						title="Research papers"
						hint="Each paper is a distinct document linked to this project — add as many as you expect to publish."
						complete={isComplete(spec, "papers")}
					>
						<NewProjectPapersStep spec={spec} onChange={setSpec} />
					</SectionShell>

					<SectionShell
						sectionId="review"
						title="Review"
						hint="Confirm the workspace bundle before creating everything."
						complete={isComplete(spec, "review")}
					>
						<Typography.Paragraph variant="muted">
							Launch creates the research project, team memberships, starter
							Kanban cards, and{" "}
							{spec.papers.length === 0
								? "no papers yet (add them later from the editor)."
								: `${spec.papers.length} linked paper${spec.papers.length === 1 ? "" : "s"}.`}
						</Typography.Paragraph>
					</SectionShell>
				</Flex.Column>

				<div className="min-h-[280px] lg:sticky lg:top-4 lg:self-start">
					<NewProjectPreview spec={spec} memberLabels={memberLabels} />
				</div>
			</div>
		</Flex.Column>
	);
};
