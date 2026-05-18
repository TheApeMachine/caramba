import { ClerkProvider } from "@clerk/tanstack-react-start";
import { auth } from "@clerk/tanstack-react-start/server";
import { TanStackDevtools } from "@tanstack/react-devtools";
import type { QueryClient } from "@tanstack/react-query";
import {
	ClientOnly,
	createRootRouteWithContext,
	HeadContent,
	redirect,
	Scripts,
} from "@tanstack/react-router";
import { TanStackRouterDevtoolsPanel } from "@tanstack/react-router-devtools";
import type React from "react";
import { Assistant } from "#/components/assistant/component";
import { AuthenticatedBoundary } from "#/components/auth/authenticated-boundary";
import { SessionControls } from "#/components/auth/session-controls";
import { Page } from "#/components/layout/page";
import { ToastProvider } from "#/components/ui/toast";
import { isAuthenticationPublicPath } from "#/lib/authentication-public-path";
import { ThemeProvider } from "#/providers/theme";
import appCss from "../styles.css?url";

const RootDocument = ({ children }: { children: React.ReactNode }) => {
	return (
		<html lang="en" suppressHydrationWarning>
			<head>
				<HeadContent />
				<script
					dangerouslySetInnerHTML={{
						__html: `(function(){try{var t=localStorage.getItem('caramba.theme')||'dark';var c=localStorage.getItem('caramba.contrast')==='1';var r=document.documentElement;['light','dim','dark'].forEach(function(x){r.classList.remove(x)});var resolved=t==='system'?(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'):t;r.classList.add(resolved);r.classList.toggle('contrast',c);}catch(e){}})();`,
					}}
				/>
			</head>
			<body className="flex h-full min-h-svh flex-col" suppressHydrationWarning>
				<ClerkProvider signInUrl="/sign-in" signUpUrl="/sign-up">
					<ThemeProvider>
						<ToastProvider>
							<Page>
								<Page.Header>
									<SessionControls />
								</Page.Header>
								<Page.Nav />
								<Page.Main>
									<Page.MainBody>
										<AuthenticatedBoundary>{children}</AuthenticatedBoundary>
									</Page.MainBody>
								</Page.Main>
								<Page.Aside>{/* reserved for layout */}</Page.Aside>
								<Page.Footer />
							</Page>
							<ClientOnly fallback={null}>
								<Assistant />
							</ClientOnly>
						</ToastProvider>
						<TanStackDevtools
							config={{
								position: "bottom-right",
							}}
							plugins={[
								{
									name: "Tanstack Router",
									render: <TanStackRouterDevtoolsPanel />,
								},
							]}
						/>
						<Scripts />
					</ThemeProvider>
				</ClerkProvider>
			</body>
		</html>
	);
};

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()(
	{
		beforeLoad: async ({ location }) => {
			if (isAuthenticationPublicPath(location.pathname)) {
				return;
			}

			if (!import.meta.env.SSR) {
				return;
			}

			const authenticationState = await auth();

			if (!authenticationState.userId) {
				throw redirect({
					to: "/sign-in",
					search:
						location.pathname !== "/" && location.pathname !== ""
							? { redirect: location.pathname }
							: {},
				});
			}
		},
		head: () => ({
			title: "caramba",
			meta: [
				{ charSet: "utf-8" },
				{ name: "viewport", content: "width=device-width, initial-scale=1" },
			],
			links: [{ rel: "stylesheet", href: appCss }],
		}),
		shellComponent: RootDocument,
		notFoundComponent: () => (
			<div className="flex h-full items-center justify-center text-muted-foreground">
				Page not found
			</div>
		),
	},
);
