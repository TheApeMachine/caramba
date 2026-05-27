/*
JazzClerkProvider — bridges Clerk identity into Jazz 2.0 external (JWT) auth.

Flow: Clerk issues a session JWT, Jazz validates it against Clerk's JWKS
(configured on the self-hosted server via --jwks-url), and the JWT's claims are
available to permissions.ts. This is the CRDT-primary, single-WebSocket data
layer that replaces the Electric shape proxies.

NOT wired into __root.tsx yet — wire it in once the migration of the data
collections off Electric is far enough along that gated surfaces have a Jazz
session to read through.

Verified against jazz-tools@2.0.0-alpha.50 (dist/react/provider.d.ts):
  - JazzProvider requires { config: DbConfig, createJazzClient, onJWTExpired? }.
    createJazzClient is exported from jazz-tools/react.
  - DbConfig: { appId, serverUrl?, jwtToken?, secret? } (jwtToken/secret are
    mutually exclusive — JWT here, local seed in the /jazz-test demo).
  - onJWTExpired: () => Promise<string | null | undefined>. Jazz calls it when
    the bearer token expires and swaps the fresh token in place (no remount,
    no full re-sync), which replaces the old fixed-interval refresher.
  - If permissions need org/role claims, mint via a Clerk JWT template:
    getToken({ template: "jazz" }).
*/

import { useAuth } from "@clerk/tanstack-react-start";
import { createJazzClient, JazzProvider } from "jazz-tools/react";
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
		<JazzProvider
			config={{ appId: APP_ID, serverUrl: SERVER_URL, jwtToken: token }}
			createJazzClient={createJazzClient}
			onJWTExpired={getToken}
		>
			{children}
		</JazzProvider>
	);
};
