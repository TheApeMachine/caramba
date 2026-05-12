import { ClerkProvider } from "@clerk/tanstack-react-start";
import { auth } from "@clerk/tanstack-react-start/server";
import { TanStackDevtools } from "@tanstack/react-devtools";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
	createRootRoute,
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
import appCss from "../styles.css?url";

const queryClient = new QueryClient();

const RootDocument = ({ children }: { children: React.ReactNode }) => {
	return (
		<html lang="en" className="dark">
			<head>
				<HeadContent />
			</head>
			<body className="flex h-full min-h-svh flex-col">
				<ClerkProvider signInUrl="/sign-in" signUpUrl="/sign-up">
					<QueryClientProvider client={queryClient}>
						<ToastProvider>
							<Page>
								<Page.Header>
									<SessionControls />
								</Page.Header>
								<Page.Nav />
								<Page.Main>
									<div className="flex h-full min-h-0 w-full min-w-0 flex-1 flex-col">
										<AuthenticatedBoundary>{children}</AuthenticatedBoundary>
									</div>
								</Page.Main>
								<Page.Aside></Page.Aside>
								<Page.Footer />
							</Page>
							<Assistant />
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
					</QueryClientProvider>
				</ClerkProvider>
			</body>
		</html>
	);
};

export const Route = createRootRoute({
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
		meta: [
			{
				charSet: "utf-8",
			},
			{
				name: "viewport",
				content: "width=device-width, initial-scale=1",
			},
			{
				title: "caramba",
			},
		],
		links: [
			{
				rel: "stylesheet",
				href: appCss,
			},
		],
	}),
	shellComponent: RootDocument,
	notFoundComponent: () => (
		<div className="flex h-full items-center justify-center text-muted-foreground">
			Page not found
		</div>
	),
});
