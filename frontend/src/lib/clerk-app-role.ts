/*
Clerk application roles via user Public metadata (Dashboard → Users → Metadata → Public).

Use JSON { "role": "admin" } for full admin in the UI. Backend parity uses
cmd/asset/config.yml clerk.admin_subject_ids until session tokens expose metadata via a JWT customization template.

Organizations: compare active org slug to VITE_CLERK_ORGANIZATION_SLUG (default caramba)
and Clerk default Admin session role (org:admin or admin).
*/
export const CLERK_PUBLIC_METADATA_ROLE_KEY = "role";

export const CLERK_APP_ROLE_ADMIN = "admin";

/*
resolvedPrivilegedOrganizationSlug is the app org slug used for UI admin checks (env overrides default).
*/
export function resolvedPrivilegedOrganizationSlug(): string {
	const envSlug = import.meta.env.VITE_CLERK_ORGANIZATION_SLUG;

	if (typeof envSlug === "string" && envSlug.trim().length > 0) {
		return envSlug.trim();
	}

	return "caramba";
}

/*
clerkBuiltInOrganizationAdminRole matches Clerk default Organization Admin role on the session.
*/
export function clerkBuiltInOrganizationAdminRole(
	activeOrganizationRole: string | null | undefined,
): boolean {
	if (activeOrganizationRole === null || activeOrganizationRole === undefined) {
		return false;
	}

	const trimmedRole = activeOrganizationRole.trim();

	return trimmedRole === "org:admin" || trimmedRole === "admin";
}

/*
privilegedOrganizationAdminMembership is true when the active org slug matches expectedSlug
(case-insensitive) and the membership role is org admin.
*/
export function privilegedOrganizationAdminMembership(
	activeOrganizationSlug: string | null | undefined,
	activeOrganizationRole: string | null | undefined,
	expectedOrganizationSlug: string,
): boolean {
	if (
		activeOrganizationSlug === null ||
		activeOrganizationSlug === undefined ||
		activeOrganizationSlug.trim() === ""
	) {
		return false;
	}

	if (
		activeOrganizationSlug.trim().toLowerCase() !==
		expectedOrganizationSlug.trim().toLowerCase()
	) {
		return false;
	}

	return clerkBuiltInOrganizationAdminRole(activeOrganizationRole);
}

/*
appRoleFromPublicMetadata reads the role string from Clerk publicMetadata.
*/
export function appRoleFromPublicMetadata(
	publicMetadata: Record<string, unknown> | undefined | null,
): string | undefined {
	if (!publicMetadata) {
		return undefined;
	}

	const roleValue = publicMetadata[CLERK_PUBLIC_METADATA_ROLE_KEY];

	if (typeof roleValue !== "string") {
		return undefined;
	}

	return roleValue;
}

/*
subjectMatchesAdminAppRole is true when role is the configured admin sentinel.
*/
export function subjectMatchesAdminAppRole(role: string | undefined): boolean {
	return role === CLERK_APP_ROLE_ADMIN;
}
