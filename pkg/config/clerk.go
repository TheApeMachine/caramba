package config

import (
	"strings"

	"github.com/spf13/viper"
)

var (
	clerkRootKey              = "clerk"
	clerkActive               = clerkRootKey + ".active"
	clerkSecret               = clerkRootKey + ".secret_key"
	clerkAdminSubjectsKey     = clerkRootKey + ".admin_subject_ids"
	clerkPrivilegedOrgSlugKey = clerkRootKey + ".privileged_organization_slug"
	clerkRequireAuthKey       = clerkRootKey + ".require_auth"
)

/*
ClerkConfig holds Clerk verification settings for the Go API.

When Active is false or secret_key is empty, /backend routes do not require a
session JWT.

AdminSubjectIDs lists Clerk user IDs (JWT sub claim values such as user_…).

PrivilegedOrganizationSlug: when non-empty, JWT claims org_slug matching this
value (case-insensitive) and org_role org:admin or admin yield clerkAdmin=true
on /backend alongside allowlisted subjects.
*/
type ClerkConfig struct {
	Active                     bool
	SecretKey                  string
	AdminSubjectIDs            []string
	PrivilegedOrganizationSlug string
	RequireAuth                bool
}

/*
adminSubjectIDsFromViper reads clerk.admin_subject_ids from YAML or viper.Set.
*/
func adminSubjectIDsFromViper() []string {
	rawIDs := viper.GetStringSlice(clerkAdminSubjectsKey)
	filteredIDs := make([]string, 0, len(rawIDs))

	for _, identifier := range rawIDs {
		trimmed := strings.TrimSpace(identifier)

		if trimmed != "" {
			filteredIDs = append(filteredIDs, trimmed)
		}
	}

	return filteredIDs
}

/*
subjectListedAsElevatedAdmin returns true when clerkSubject is allowlisted.
*/
func (clerkConfig *ClerkConfig) subjectListedAsElevatedAdmin(clerkSubject string) bool {
	if strings.TrimSpace(clerkSubject) == "" {
		return false
	}

	for _, adminSubjectID := range clerkConfig.AdminSubjectIDs {
		if adminSubjectID == clerkSubject {
			return true
		}
	}

	return false
}

/*
sessionClaimsPrivilegedOrganizationAdmin reflects Clerk Organizations: active org
slug matches configured privileged slug and role is the built-in Admin mapping.
*/
func (clerkConfig *ClerkConfig) sessionClaimsPrivilegedOrganizationAdmin(
	activeOrganizationSlug string,
	activeOrganizationRole string,
) bool {
	configSlug := strings.TrimSpace(clerkConfig.PrivilegedOrganizationSlug)

	if configSlug == "" {
		return false
	}

	sessionSlug := strings.TrimSpace(activeOrganizationSlug)

	if sessionSlug == "" || !strings.EqualFold(sessionSlug, configSlug) {
		return false
	}

	switch strings.TrimSpace(activeOrganizationRole) {
	case "org:admin", "admin":
		return true
	default:
		return false
	}
}

/*
SubjectHasElevatedAdminPrivileges returns true for allowlisted JWT subjects or
for active organization admin on the privileged org slug.
*/
func (clerkConfig *ClerkConfig) SubjectHasElevatedAdminPrivileges(
	clerkSubject string,
	activeOrganizationSlug string,
	activeOrganizationRole string,
) bool {
	if clerkConfig.subjectListedAsElevatedAdmin(clerkSubject) {
		return true
	}

	return clerkConfig.sessionClaimsPrivilegedOrganizationAdmin(
		activeOrganizationSlug,
		activeOrganizationRole,
	)
}

/*
NewClerkConfig reads Clerk settings from viper-loaded config.yml.
*/
func NewClerkConfig() *ClerkConfig {
	return &ClerkConfig{
		Active:                     WithDefault(clerkActive, true),
		SecretKey:                  WithDefault(clerkSecret, ""),
		AdminSubjectIDs:            adminSubjectIDsFromViper(),
		PrivilegedOrganizationSlug: WithDefault(clerkPrivilegedOrgSlugKey, ""),
		RequireAuth:                WithDefault(clerkRequireAuthKey, true),
	}
}
