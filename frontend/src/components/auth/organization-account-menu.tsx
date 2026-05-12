import {
	useOrganization,
	useOrganizationList,
} from "@clerk/tanstack-react-start";
import { CheckIcon, ChevronDownIcon } from "lucide-react";
import { Button } from "#/components/ui/button";
import {
	Menu,
	MenuItem,
	MenuPopup,
	MenuSeparator,
	MenuTrigger,
} from "#/components/ui/menu";

/*
OrganizationAccountMenu switches between the user's personal workspace and Clerk organizations,
without duplicating the avatar-driven dropdown rendered by UserButton.
*/
export function OrganizationAccountMenu() {
	const { organization, isLoaded: organizationLoaded } = useOrganization();
	const {
		isLoaded: membershipListLoaded,
		setActive,
		userMemberships,
	} = useOrganizationList({
		userMemberships: {
			infinite: true,
		},
	});

	const loaded = organizationLoaded && membershipListLoaded;

	const memberships = userMemberships?.data ?? [];
	const label =
		organization !== undefined && organization !== null
			? organization.name
			: "Personal account";

	return (
		<Menu>
			<MenuTrigger
				render={
					<Button
						className="max-w-[min(100vw-8rem,14rem)] gap-2 text-foreground"
						disabled={!loaded}
						size="sm"
						type="button"
						variant="outline"
					/>
				}
				aria-label="Switch workspace"
			>
				<span className="min-w-0 flex-1 truncate text-start" title={label}>
					{label}
				</span>
				<ChevronDownIcon aria-hidden className="size-4 shrink-0 opacity-80" />
			</MenuTrigger>
			<MenuPopup align="end" className="min-w-48">
				<MenuItem
					disabled={!loaded}
					onClick={() => {
						void setActive?.({ organization: null });
					}}
				>
					<span className="flex min-w-0 flex-1 items-center gap-2">
						<span className="min-w-0 flex-1 truncate">Personal account</span>
						{loaded && organization === null ? (
							<CheckIcon aria-hidden className="size-4 shrink-0" />
						) : null}
					</span>
				</MenuItem>
				{memberships.length > 0 ? <MenuSeparator /> : null}
				{memberships.map((membership) => {
					const activeOrganization =
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
							}}
						>
							<span className="flex min-w-0 flex-1 items-center gap-2">
								<span
									className="min-w-0 flex-1 truncate"
									title={membership.organization.name}
								>
									{membership.organization.name}
								</span>
								{activeOrganization ? (
									<CheckIcon aria-hidden className="size-4 shrink-0" />
								) : null}
							</span>
						</MenuItem>
					);
				})}
				{userMemberships?.hasNextPage ? (
					<>
						<MenuSeparator />
						<MenuItem
							disabled={userMemberships.isFetching}
							onClick={() => {
								void userMemberships.fetchNext();
							}}
						>
							Load more organizations
						</MenuItem>
					</>
				) : null}
			</MenuPopup>
		</Menu>
	);
}
