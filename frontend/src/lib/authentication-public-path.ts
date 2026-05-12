/*
isAuthenticationPublicPath returns true for routes that must load without enforcing a Clerk session in navigation guards.
*/
export function isAuthenticationPublicPath(pathname: string): boolean {
	if (pathname.startsWith("/sign-in")) {
		return true;
	}

	if (pathname.startsWith("/sign-up")) {
		return true;
	}

	if (pathname.startsWith("/api/")) {
		return true;
	}

	return false;
}
