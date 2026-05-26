"use client";

import {
	useAuth,
	useOrganization,
	useOrganizationList,
} from "@clerk/tanstack-react-start";
import { useLiveQuery } from "@tanstack/react-db";
import { useForm } from "@tanstack/react-form";
import { ClientOnly, useNavigate } from "@tanstack/react-router";
import {
	BuildingIcon,
	CheckIcon,
	ChevronDownIcon,
	PlusIcon,
	UsersIcon,
} from "lucide-react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { teamCollection } from "#/collections/team";
import { Button } from "#/components/ui/button";
import {
	Dialog,
	DialogClose,
	DialogFooter,
	DialogHeader,
	DialogPanel,
	DialogPopup,
	DialogTitle,
} from "#/components/ui/dialog";
import { Field } from "#/components/ui/field";
import { Flex } from "#/components/ui/flex";
import { Input } from "#/components/ui/input";
import {
	Menu,
	MenuItem,
	MenuPopup,
	MenuSeparator,
	MenuTrigger,
} from "#/components/ui/menu";
import { Typography } from "#/components/ui/typography";
import { setActiveTeam, useActiveTeam } from "#/lib/active-team";
import { deriveSlug } from "#/lib/derive-slug";

/*
WorkspaceSwitcher replaces the old org-only menu with a combined
org + team picker. Section 1: Clerk organizations (and personal).
Section 2: teams inside the active org, synced via Electric.
*/
const WorkspaceSwitcherFallback = () => {
	const { t } = useTranslation();

	return (
		<Button
			aria-label={t("auth.switchWorkspace")}
			className="max-w-[min(100vw-8rem,18rem)] gap-2 text-foreground"
			disabled
			size="sm"
			type="button"
			variant="outline"
		>
			<BuildingIcon aria-hidden className="size-4 shrink-0 opacity-80" />
			<span className="min-w-0 flex-1 truncate text-start opacity-60">…</span>
			<ChevronDownIcon aria-hidden className="size-4 shrink-0 opacity-80" />
		</Button>
	);
};

const WorkspaceSwitcherContent = () => {
	const { t } = useTranslation();
	const { orgId } = useAuth();
	const { organization, isLoaded: organizationLoaded } = useOrganization();
	const {
		isLoaded: membershipListLoaded,
		setActive,
		userMemberships,
	} = useOrganizationList({
		userMemberships: { infinite: true },
	});

	const activeTeamId = useActiveTeam(orgId);
	const [createOpen, setCreateOpen] = useState(false);

	const loaded = organizationLoaded && membershipListLoaded;
	const memberships = userMemberships?.data ?? [];

	const { data: teams } = useLiveQuery((query) =>
		query.from({ team: teamCollection }).select(({ team }) => ({
			id: team.id,
			name: team.name,
			slug: team.slug,
			organization_slug: team.organization_slug,
		})),
	);

	const activeTeam = useMemo(
		() => (teams ?? []).find((team) => team.id === activeTeamId) ?? null,
		[teams, activeTeamId],
	);

	const orgLabel = organization?.name ?? t("auth.personalAccount");
	const triggerLabel = activeTeam
		? `${orgLabel} · ${activeTeam.name}`
		: orgLabel;

	return (
		<Flex.Row className="items-center gap-2">
			<Menu>
				<MenuTrigger
					aria-label={t("auth.switchWorkspace")}
					render={
						<Button
							className="max-w-[min(100vw-8rem,18rem)] gap-2 text-foreground"
							disabled={!loaded}
							size="sm"
							type="button"
							variant="outline"
						/>
					}
				>
					<BuildingIcon aria-hidden className="size-4 shrink-0 opacity-80" />
					<span
						className="min-w-0 flex-1 truncate text-start"
						title={triggerLabel}
					>
						{triggerLabel}
					</span>
					<ChevronDownIcon aria-hidden className="size-4 shrink-0 opacity-80" />
				</MenuTrigger>
				<MenuPopup align="end" className="min-w-64">
					<Typography.Span
						className="px-2 pt-1.5 pb-0.5 text-[10px] font-medium uppercase tracking-wider"
						variant="muted"
					>
						{t("auth.organization", { defaultValue: "Organization" })}
					</Typography.Span>
					<MenuItem
						disabled={!loaded}
						onClick={() => {
							void setActive?.({ organization: null });
							setActiveTeam(null, null);
						}}
					>
						<OrgOrTeamRow
							label={t("auth.personalAccount")}
							selected={loaded && organization === null}
						/>
					</MenuItem>
					{memberships.map((membership) => {
						const isActive =
							organization !== undefined &&
							organization !== null &&
							organization.id === membership.organization.id;

						return (
							<MenuItem
								key={membership.id}
								onClick={() => {
									void setActive?.({
										organization: membership.organization.id,
									});
									setActiveTeam(membership.organization.id, null);
								}}
							>
								<OrgOrTeamRow
									label={membership.organization.name}
									selected={isActive}
								/>
							</MenuItem>
						);
					})}

					<MenuSeparator />

					<Typography.Span
						className="px-2 pt-1.5 pb-0.5 text-[10px] font-medium uppercase tracking-wider"
						variant="muted"
					>
						{t("auth.teams", { defaultValue: "Teams" })}
					</Typography.Span>
					<MenuItem onClick={() => setActiveTeam(orgId, null)}>
						<OrgOrTeamRow
							label={t("auth.allTeams", { defaultValue: "All teams" })}
							selected={activeTeamId === null}
						/>
					</MenuItem>
					{(teams ?? []).map((team) => (
						<MenuItem
							key={team.id}
							onClick={() => setActiveTeam(orgId, team.id)}
						>
							<OrgOrTeamRow
								label={team.name}
								selected={activeTeamId === team.id}
							/>
						</MenuItem>
					))}
					{(teams ?? []).length === 0 ? (
						<Flex.Row className="px-2 py-1.5">
							<Typography.Span variant="muted" className="text-xs">
								{t("auth.noTeams", { defaultValue: "No teams yet." })}
							</Typography.Span>
						</Flex.Row>
					) : null}

					<MenuSeparator />

					<MenuItem onClick={() => setCreateOpen(true)}>
						<Flex.Row className="items-center gap-2">
							<PlusIcon aria-hidden className="size-4" />
							<Typography.Span>
								{t("auth.createTeam", { defaultValue: "Create team…" })}
							</Typography.Span>
						</Flex.Row>
					</MenuItem>

					{userMemberships?.hasNextPage ? (
						<>
							<MenuSeparator />
							<MenuItem
								disabled={userMemberships.isFetching}
								onClick={() => void userMemberships.fetchNext()}
							>
								{t("auth.loadMoreOrganizations")}
							</MenuItem>
						</>
					) : null}
				</MenuPopup>
			</Menu>

			<CreateTeamDialog
				onOpenChange={setCreateOpen}
				open={createOpen}
				organizationSlug={
					organization?.slug ?? memberships[0]?.organization?.slug ?? ""
				}
			/>
		</Flex.Row>
	);
};

export const WorkspaceSwitcher = () => (
	<ClientOnly fallback={<WorkspaceSwitcherFallback />}>
		<WorkspaceSwitcherContent />
	</ClientOnly>
);

const OrgOrTeamRow = ({
	label,
	selected,
}: {
	label: string;
	selected: boolean;
}) => {
	return (
		<Flex.Row className="min-w-0 flex-1 items-center gap-2">
			<UsersIcon aria-hidden className="size-3.5 shrink-0 opacity-70" />
			<span className="min-w-0 flex-1 truncate" title={label}>
				{label}
			</span>
			{selected ? <CheckIcon aria-hidden className="size-4 shrink-0" /> : null}
		</Flex.Row>
	);
};

const CreateTeamDialog = ({
	open,
	onOpenChange,
	organizationSlug,
}: {
	open: boolean;
	onOpenChange: (next: boolean) => void;
	organizationSlug: string;
}) => {
	const { orgId } = useAuth();
	const { t } = useTranslation();
	const navigate = useNavigate();
	const [submitError, setSubmitError] = useState<string | null>(null);

	const form = useForm({
		defaultValues: { name: "" },
		onSubmit: async ({ value, formApi }) => {
			const trimmed = value.name.trim();
			setSubmitError(null);

			try {
				const id = crypto.randomUUID();
				const slug = deriveSlug(trimmed);
				const transaction = teamCollection.insert({
					id,
					organization_slug: organizationSlug,
					name: trimmed,
					slug,
					description: "",
					created_at: new Date(),
					updated_at: new Date(),
				});

				await transaction.isPersisted.promise;
				setActiveTeam(orgId, id);
				formApi.reset();
				onOpenChange(false);
				navigate({
					to: "/$orgSlug/$teamSlug/setup",
					params: { orgSlug: organizationSlug, teamSlug: slug },
				});
			} catch (err) {
				setSubmitError(err instanceof Error ? err.message : String(err));
			}
		},
	});

	return (
		<Dialog onOpenChange={onOpenChange} open={open}>
			<DialogPopup>
				<DialogHeader>
					<DialogTitle>
						{t("auth.createTeamTitle", { defaultValue: "Create team" })}
					</DialogTitle>
				</DialogHeader>
				<form
					onSubmit={(event) => {
						event.preventDefault();
						event.stopPropagation();
						void form.handleSubmit();
					}}
				>
					<DialogPanel>
						<Flex.Column className="gap-3">
							<form.Field
								name="name"
								validators={{
									onChange: ({ value }) =>
										value.trim().length < 1
											? t("auth.teamNameRequired", {
													defaultValue: "Name is required.",
												})
											: undefined,
								}}
							>
								{(field) => (
									<Field
										data-invalid={
											field.state.meta.isTouched && !field.state.meta.isValid
										}
									>
										<Field.Label htmlFor={field.name}>
											{t("auth.teamName", { defaultValue: "Team name" })}
										</Field.Label>
										<Input
											aria-invalid={
												field.state.meta.isTouched && !field.state.meta.isValid
											}
											autoFocus
											id={field.name}
											name={field.name}
											onBlur={field.handleBlur}
											onChange={(event) =>
												field.handleChange(event.target.value)
											}
											placeholder={t("auth.teamNamePlaceholder", {
												defaultValue: "e.g. Architecture",
											})}
											value={field.state.value}
										/>
										{field.state.meta.isTouched &&
										field.state.meta.errors.length ? (
											<Field.Error>
												{field.state.meta.errors.join(", ")}
											</Field.Error>
										) : null}
									</Field>
								)}
							</form.Field>
							{submitError ? (
								<Typography.Span variant="error" className="text-xs">
									{submitError}
								</Typography.Span>
							) : null}
						</Flex.Column>
					</DialogPanel>
					<DialogFooter>
						<DialogClose
							render={
								<Button type="button" variant="ghost">
									{t("common.cancel", { defaultValue: "Cancel" })}
								</Button>
							}
						/>
						<form.Subscribe
							selector={(state) =>
								[state.canSubmit, state.isSubmitting] as const
							}
						>
							{([canSubmit, isSubmitting]) => (
								<Button disabled={!canSubmit || isSubmitting} type="submit">
									{isSubmitting
										? t("auth.creatingTeam", { defaultValue: "Creating…" })
										: t("auth.createTeamSubmit", {
												defaultValue: "Create team",
											})}
								</Button>
							)}
						</form.Subscribe>
					</DialogFooter>
				</form>
			</DialogPopup>
		</Dialog>
	);
};
