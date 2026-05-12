import { useAuth } from "@clerk/tanstack-react-start";
import { Navigate, useRouterState } from "@tanstack/react-router";
import type React from "react";
import { isAuthenticationPublicPath } from "#/lib/authentication-public-path";

/*
AuthenticatedBoundary mirrors root-route auth for client navigations.
Server-side auth() reads TanStack Start request context; during SPA transitions that context is absent, so guards must run here instead.
*/
export function AuthenticatedBoundary({
	children,
}: {
	children: React.ReactNode;
}) {
	const pathname = useRouterState({
		select: (routerState) => routerState.location.pathname,
	});
	const { isSignedIn, isLoaded } = useAuth();

	if (isAuthenticationPublicPath(pathname)) {
		return children;
	}

	if (!isLoaded) {
		return (
			<div className="flex h-full flex-1 items-center justify-center text-muted-foreground">
				Loading session…
			</div>
		);
	}

	if (!isSignedIn) {
		const redirectSearch =
			pathname !== "/" && pathname !== ""
				? ({ redirect: pathname } as const)
				: {};

		return <Navigate replace search={redirectSearch} to="/sign-in" />;
	}

	return children;
}
