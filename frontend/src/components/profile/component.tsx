"use client";

import type { useUser } from "@clerk/tanstack-react-start";
import { PencilIcon } from "lucide-react";
import {
	nextProfileExpansion,
	type ProfileDraft,
	type ProfileExpansion,
	profileDescriptionLine,
	profileDisplayLabel,
	profileInitials,
	profileSpring,
} from "#/components/profile/model";
import { ProfileForm } from "#/components/profile/profile-form";
import { ProfileReadout } from "#/components/profile/profile-readout";
import { Avatar, AvatarFallback, AvatarImage } from "#/components/ui/avatar";
import { Button } from "#/components/ui/button";
import { AnimatePresence, Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";
import { cn } from "#/lib/utils";

export type {
	ProfileDraft,
	ProfileExpansion,
} from "#/components/profile/model";

interface ProfileComponentProps {
	user: NonNullable<ReturnType<typeof useUser>["user"]>;
	expansion: ProfileExpansion;
	onExpansionChange?: (expansion: ProfileExpansion) => void;
	draft: ProfileDraft;
	onDraftChange: (draft: ProfileDraft) => void;
	description?: string;
	error?: string | null;
	saving?: boolean;
	onSave?: () => void;
	onCancelEdit?: () => void;
	showEditControl?: boolean;
	className?: string;
}

const avatarSizeClass = (expansion: ProfileExpansion): string => {
	if (expansion === "avatar") {
		return "size-12";
	}

	if (expansion === "identity") {
		return "size-10";
	}

	return "size-14";
};

export const ProfileComponent = ({
	user,
	expansion,
	onExpansionChange,
	draft,
	onDraftChange,
	description = "How you show up on cards, papers, and collaboration.",
	error,
	saving = false,
	onSave,
	onCancelEdit,
	showEditControl = false,
	className,
}: ProfileComponentProps) => {
	const displayLabel = profileDisplayLabel(draft);
	const initials = profileInitials(draft);
	const email = user.primaryEmailAddress?.emailAddress ?? "";
	const descriptionLine =
		profileDescriptionLine(draft, description) || email || description;

	const showIdentity = expansion !== "avatar";
	const showSummary = expansion === "summary" || expansion === "form";
	const showForm = expansion === "form";
	const showReadout = expansion === "summary";

	const handleShellClick = () => {
		if (!onExpansionChange) {
			return;
		}

		const next = nextProfileExpansion(expansion);

		if (next) {
			onExpansionChange(next);
		}
	};

	const handleEditToggle = () => {
		if (!showEditControl || !onExpansionChange) {
			return;
		}

		if (expansion === "form") {
			onCancelEdit?.();
			onExpansionChange("summary");
			return;
		}

		onExpansionChange("form");
	};

	return (
		<Flex
			layout
			transition={profileSpring}
			className={cn(
				"overflow-hidden border border-border/60 bg-card/90 shadow-sm backdrop-blur-md",
				showSummary && showEditControl && "group relative",
				expansion === "avatar" &&
					"inline-flex size-12 cursor-pointer items-center justify-center rounded-full",
				expansion === "identity" &&
					"inline-flex max-w-xs cursor-pointer items-center gap-2 rounded-full px-1 py-1",
				(expansion === "summary" || expansion === "form") &&
					"flex w-full min-h-0 flex-col gap-3 rounded-2xl p-3",
				className,
			)}
			onClick={expansion === "form" ? undefined : handleShellClick}
			role={expansion === "form" ? undefined : "button"}
			tabIndex={expansion === "form" ? undefined : 0}
			onKeyDown={(event) => {
				if (expansion === "form" || !onExpansionChange) {
					return;
				}

				if (event.key === "Enter" || event.key === " ") {
					event.preventDefault();
					handleShellClick();
				}
			}}
			aria-label={
				expansion === "avatar"
					? "Expand profile"
					: expansion === "identity"
						? "Expand profile details"
						: undefined
			}
		>
			<Flex.Row
				layout
				align={expansion === "avatar" ? "center" : "start"}
				gap={expansion === "identity" ? 2 : 3}
				className={cn(
					"min-w-0",
					expansion === "avatar" && "size-12 justify-center",
					showSummary && "w-full",
				)}
			>
				<Flex
					layout="position"
					layoutId="profile-avatar"
					transition={profileSpring}
					className="shrink-0"
				>
					<Avatar
						className={cn(
							"border-2 border-primary/20",
							avatarSizeClass(expansion),
						)}
					>
						{user.imageUrl ? <AvatarImage alt="" src={user.imageUrl} /> : null}
						<AvatarFallback
							className={cn(expansion === "avatar" ? "text-sm" : "text-base")}
						>
							{initials}
						</AvatarFallback>
					</Avatar>
				</Flex>

				<AnimatePresence mode="popLayout" initial={false}>
					{showIdentity ? (
						<Flex.Column
							key="profile-identity"
							layout
							initial={{ opacity: 0, x: -8 }}
							animate={{ opacity: 1, x: 0 }}
							exit={{ opacity: 0, x: -8 }}
							transition={profileSpring}
							gap={1}
							className="min-w-0 flex-1"
						>
							<Typography.H4
								variant="sectionHeading"
								className={cn(
									"truncate",
									expansion === "identity" && "text-sm",
								)}
							>
								{displayLabel}
							</Typography.H4>

							<AnimatePresence mode="popLayout" initial={false}>
								{showSummary ? (
									<Flex.Column
										key="profile-summary-copy"
										layout
										initial={{ opacity: 0, height: 0 }}
										animate={{ opacity: 1, height: "auto" }}
										exit={{ opacity: 0, height: 0 }}
										transition={profileSpring}
										gap={1}
									>
										<Typography.Paragraph
											variant="muted"
											className="text-pretty text-xs leading-snug sm:text-sm"
										>
											{descriptionLine}
										</Typography.Paragraph>
										{email && expansion === "summary" ? (
											<Typography.Small variant="muted" className="truncate">
												Account: {email}
											</Typography.Small>
										) : null}
									</Flex.Column>
								) : null}
							</AnimatePresence>
						</Flex.Column>
					) : null}
				</AnimatePresence>

				{showSummary && showEditControl ? (
					<Button
						type="button"
						variant={showForm ? "outline" : "ghost"}
						size="icon-sm"
						className={cn(
							"shrink-0 opacity-0 transition-opacity duration-200",
							showForm
								? "opacity-100"
								: "pointer-events-none group-hover:pointer-events-auto group-hover:opacity-100 group-focus-within:pointer-events-auto group-focus-within:opacity-100",
						)}
						aria-label={showForm ? "Cancel editing profile" : "Edit profile"}
						onClick={(event) => {
							event.stopPropagation();
							handleEditToggle();
						}}
					>
						<PencilIcon className="size-4" />
					</Button>
				) : null}
			</Flex.Row>

			<AnimatePresence mode="popLayout" initial={false}>
				{error ? (
					<Flex
						key="profile-error"
						layout
						initial={{ opacity: 0, height: 0 }}
						animate={{ opacity: 1, height: "auto" }}
						exit={{ opacity: 0, height: 0 }}
						transition={profileSpring}
						fullWidth
					>
						<Typography.Paragraph className="text-destructive text-xs">
							{error}
						</Typography.Paragraph>
					</Flex>
				) : null}
			</AnimatePresence>

			<AnimatePresence mode="wait" initial={false}>
				{showForm && onSave ? (
					<Flex
						key="profile-form"
						layout
						initial={{ opacity: 0, y: 12 }}
						animate={{ opacity: 1, y: 0 }}
						exit={{ opacity: 0, y: 8 }}
						transition={profileSpring}
						fullWidth
						className="min-h-0 flex-1"
					>
						<ProfileForm
							draft={draft}
							saving={saving}
							onDraftChange={onDraftChange}
							onSave={onSave}
						/>
					</Flex>
				) : null}

				{showReadout ? (
					<Flex
						key="profile-readout"
						layout
						initial={{ opacity: 0, y: 8 }}
						animate={{ opacity: 1, y: 0 }}
						exit={{ opacity: 0, y: -8 }}
						transition={profileSpring}
						fullWidth
						className="min-h-0 flex-1"
					>
						<ProfileReadout draft={draft} />
					</Flex>
				) : null}
			</AnimatePresence>
		</Flex>
	);
};
