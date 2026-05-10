import { TanStackDevtools } from "@tanstack/react-devtools";
import { createRootRoute, HeadContent, Scripts } from "@tanstack/react-router";
import { TanStackRouterDevtoolsPanel } from "@tanstack/react-router-devtools";
import type React from "react";
import { Page } from "#/components/layout/page";
import { ToastProvider } from "#/components/ui/toast";
import appCss from "../styles.css?url";

const RootDocument = ({ children }: { children: React.ReactNode }) => {
	return (
		<html lang="en" className="dark">
			<head>
				<HeadContent />
			</head>
			<body className="flex h-full min-h-svh flex-col">
				<ToastProvider>
					<Page>
						<Page.Header></Page.Header>
						<Page.Nav />
						<Page.Main>
							<div className="flex h-full min-h-0 w-full min-w-0 flex-1 flex-col">
								{children}
							</div>
						</Page.Main>
						<Page.Aside></Page.Aside>
						<Page.Footer />
					</Page>
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
			</body>
		</html>
	);
}

export const Route = createRootRoute({
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
});