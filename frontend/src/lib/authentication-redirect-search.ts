import { z } from "zod";

/*
authenticationRedirectSchema restricts post-auth redirects to same-origin paths (no open redirects).
*/
export const authenticationRedirectSchema = z.object({
	redirect: z
		.string()
		.optional()
		.refine(
			(value) => value === undefined || value.startsWith("/"),
			"redirect must be a relative path",
		),
});

export type AuthenticationRedirectSearch = z.infer<
	typeof authenticationRedirectSchema
>;

/*
validateAuthenticationRedirect parses TanStack Router search for sign-in/up routes.
*/
export function validateAuthenticationRedirect(
	rawSearch: Record<string, unknown>,
): AuthenticationRedirectSearch {
	const parsed = authenticationRedirectSchema.safeParse(rawSearch);

	return parsed.success ? parsed.data : { redirect: undefined };
}
