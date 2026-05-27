/*
JazzClerkProvider — bridges Clerk identity into Jazz 2.0 external (JWT) auth.

Flow: Clerk issues a session JWT, Jazz validates it against Clerk's JWKS
(configured on the self-hosted server via --jwks-url), and the JWT's claims are
available to permissions.ts. This is the CRDT-primary, single-WebSocket data
layer that replaces the Electric shape proxies.

Wired into __root.tsx as a child of AuthenticatedBoundary, so it only mounts
for a signed-in session. Public and signed-out surfaces pass straight through
(they never touch Jazz); a signed-in session with an unresolved bearer token
holds behind a pending state rather than rendering Jazz consumers without a
provider (which throws "useDb must be used within <JazzProvider>").

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
// Resolve the WASM binary through Vite's asset pipeline (?url) and hand it to
// Jazz explicitly via runtimeSources.wasmUrl. Without this, wasm-bindgen falls
// back to `new URL('jazz_wasm_bg.wasm', import.meta.url)`, which this toolchain
// (rolldown-vite + nitro) does not rewrite to a served asset — the fetch 404s
// and WebAssembly.compile throws "HTTP status code is not ok". The server (SSR)
// path reads the binary from disk and ignores this URL.
import jazzWasmUrl from "jazz-wasm/pkg/jazz_wasm_bg.wasm?url";
import { type ReactNode, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

const APP_ID = import.meta.env.VITE_JAZZ_APP_ID as string;
const SERVER_URL = import.meta.env.VITE_JAZZ_SERVER_URL as string;

export const JazzClerkProvider = ({ children }: { children: ReactNode }) => {
	const { isLoaded, isSignedIn, getToken } = useAuth();
	const { t } = useTranslation();
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

	// Public and signed-out surfaces never touch Jazz, so pass them straight
	// through. Only a signed-in session needs the CRDT data layer.
	if (!isLoaded || !isSignedIn) {
		return <>{children}</>;
	}

	// Signed in, but the bearer token has not resolved yet. Hold the gated
	// surfaces behind a pending state rather than rendering Jazz consumers
	// without a provider (useDb/useAll throw outside <JazzProvider>).
	if (!token) {
		return (
			<div className="flex h-full flex-1 items-center justify-center text-muted-foreground">
				{t("common.loadingSession")}
			</div>
		);
	}

	return (
		<JazzProvider
			config={{
				appId: APP_ID,
				serverUrl: SERVER_URL,
				jwtToken: token,
				runtimeSources: { wasmUrl: jazzWasmUrl },
			}}
			createJazzClient={createJazzClient}
			onJWTExpired={getToken}
		>
			{children}
		</JazzProvider>
	);
};
