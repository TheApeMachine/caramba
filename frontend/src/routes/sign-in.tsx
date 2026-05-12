import { SignIn } from "@clerk/tanstack-react-start";
import { createFileRoute } from "@tanstack/react-router";
import { validateAuthenticationRedirect } from "#/lib/authentication-redirect-search";

export const Route = createFileRoute("/sign-in")({
	validateSearch: validateAuthenticationRedirect,
	component: SignInRouteComponent,
});

function SignInRouteComponent() {
	const searchState = Route.useSearch();
	const fallbackRedirectUrl =
		searchState.redirect !== undefined && searchState.redirect.length > 0
			? searchState.redirect
			: "/";

	return (
		<div className="flex min-h-[min(640px,70vh)] w-full flex-1 items-center justify-center p-6">
			<SignIn
				fallbackRedirectUrl={fallbackRedirectUrl}
				path="/sign-in"
				routing="path"
				signUpUrl="/sign-up"
			/>
		</div>
	);
}
