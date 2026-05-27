/*
JazzClerkProvider — bridges Clerk identity into Jazz 2.0 external (JWT) auth.

Flow: Clerk issues a session JWT, Jazz validates it against Clerk's JWKS
(configured on the self-hosted server via --jwks-url), and the JWT's claims are
available to permissions.ts. This is the CRDT-primary, single-WebSocket data
layer that replaces the Electric shape proxies.

NOT wired into __root.tsx yet — wire it in after `pnpm install` + adding
jazzPlugin() to vite.config.ts, so an uninstalled import can't break dev.

Confidence:
  - JazzProvider + config { appId, serverUrl, jwtToken } : from docs (high).
  - Clerk getToken() for the JWT : standard Clerk (high). If org claims are
    needed in permissions, mint with a JWT template: getToken({ template: "jazz" }).
  - Token REFRESH (Clerk tokens are short-lived): docs show
    `db.updateAuthToken(freshJwt)` for same-principal refresh, but the React hook
    that hands you `db` is unverified here. Marked TODO below — confirm the hook
    name against the installed jazz-tools@alpha (likely useDb()/useJazzContext()).
*/

import { useAuth } from "@clerk/tanstack-react-start";
import { JazzProvider } from "jazz-tools/react";
import { type ReactNode, useEffect, useState } from "react";

const APP_ID = import.meta.env.VITE_JAZZ_APP_ID as string;
const SERVER_URL = import.meta.env.VITE_JAZZ_SERVER_URL as string;

export const JazzClerkProvider = ({ children }: { children: ReactNode }) => {
	const { isLoaded, isSignedIn, getToken } = useAuth();
	const [token, setToken] = useState<string | null>(null);

	useEffect(() => {
		if (!isLoaded || !isSignedIn) {
			return;
		}

		let cancelled = false;

		// Use a Clerk JWT template named "jazz" if/when permissions need org claims.
		getToken().then((jwt) => {
			if (!cancelled) {
				setToken(jwt);
			}
		});

		return () => {
			cancelled = true;
		};
	}, [isLoaded, isSignedIn, getToken]);

	// Render children without a Jazz session until Clerk + token resolve. Reads
	// still work for anything not gated; gated surfaces show their own pending UI.
	if (!isLoaded || !token) {
		return <>{children}</>;
	}

	return (
		<JazzProvider config={{ appId: APP_ID, serverUrl: SERVER_URL, jwtToken: token }}>
			{children}
			{/*
			TODO(verify alpha): mount a small child here that holds the `db` handle
			and calls db.updateAuthToken(await getToken()) on a timer / Clerk token
			change, so the short-lived Clerk JWT is refreshed in place instead of
			remounting the provider (which would force a full re-sync).
			*/}
		</JazzProvider>
	);
};
