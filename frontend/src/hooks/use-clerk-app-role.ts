import { useAuth, useUser } from "@clerk/tanstack-react-start";
import { useMemo } from "react";
import {
	appRoleFromPublicMetadata,
	privilegedOrganizationAdminMembership,
	resolvedPrivilegedOrganizationSlug,
	subjectMatchesAdminAppRole,
} from "#/lib/clerk-app-role";

/*
useClerkAppRole derives the application role from Clerk user.publicMetadata.
*/
export function useClerkAppRole(): {
	appRole: string | undefined;
	isLoaded: boolean;
	isSignedIn: boolean;
} {
	const { user, isLoaded } = useUser();

	return useMemo(() => {
		const appRole =
			user !== undefined && user !== null
				? appRoleFromPublicMetadata(
						user.publicMetadata as Record<string, unknown>,
					)
				: undefined;

		return {
			appRole,
			isLoaded,
			isSignedIn: user !== undefined && user !== null,
		};
	}, [user, isLoaded]);
}

/*
useIsClerkAppAdmin is true when public metadata role is admin, or when the user is
Organization Admin on the configured privileged org slug (default caramba).
*/
export function useIsClerkAppAdmin(): boolean {
	const { appRole, isLoaded: userLoaded, isSignedIn } = useClerkAppRole();
	const { orgSlug, orgRole, userId, isLoaded: authLoaded } = useAuth();

	const metadataAdministrator =
		userLoaded && isSignedIn && subjectMatchesAdminAppRole(appRole);

	const organizationAdministrator =
		authLoaded &&
		userId !== null &&
		userId !== undefined &&
		privilegedOrganizationAdminMembership(
			orgSlug,
			orgRole,
			resolvedPrivilegedOrganizationSlug(),
		);

	return metadataAdministrator || organizationAdministrator;
}
