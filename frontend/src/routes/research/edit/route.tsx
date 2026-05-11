import {
	createFileRoute,
	Outlet,
	useNavigate,
	useRouterState,
} from "@tanstack/react-router";
import { BoxIcon, HouseIcon, PanelsTopLeftIcon } from "lucide-react";
import { z } from "zod";
import { Flex } from "#/components/ui/flex";
import { Tabs } from "#/components/ui/tabs";

const editSearchSchema = z.object({
	projectId: z.uuid().optional(),
});

function parseEditSearch(
	raw: Record<string, unknown>,
): z.infer<typeof editSearchSchema> {
	const parsed = editSearchSchema.safeParse(raw);
	return parsed.success ? parsed.data : { projectId: undefined };
}

const researchEditTabPaths = {
	architecture: "/research/edit",
	"model-scope": "/research/edit/model-scope",
	"research-paper": "/research/edit/research-paper",
} as const;

type ResearchEditTabId = keyof typeof researchEditTabPaths;

function researchEditTabFromPathname(pathname: string): ResearchEditTabId {
	if (pathname.endsWith("/model-scope")) {
		return "model-scope";
	}
	if (pathname.endsWith("/research-paper")) {
		return "research-paper";
	}
	return "architecture";
}

export const Route = createFileRoute("/research/edit")({
	validateSearch: parseEditSearch,
	component: ResearchEditLayout,
});

function ResearchEditLayout() {
	const search = Route.useSearch();
	const pathname = useRouterState({
		select: (routerState) => routerState.location.pathname,
	});
	const navigate = useNavigate();
	const activeTab = researchEditTabFromPathname(pathname);

	// Expose the active tab as page context so the assistant knows where the user is.
	const activeTabLabel = { architecture: "Research graph", "model-scope": "Model Scope", "research-paper": "Research Paper" }[activeTab];

	return (
		<Flex.Column className="min-h-0" fullHeight fullWidth>
			<Tabs
				value={activeTab}
				className="items-center"
				onValueChange={(value) => {
					const targetTo = researchEditTabPaths[value as ResearchEditTabId];
					if (!targetTo) {
						return;
					}
					void navigate({ to: targetTo, search });
				}}
			>
				<div className="border-b">
					<span
						className="sr-only"
						data-context="Active tab"
						data-context-key="active_tab"
					>
						{activeTabLabel}
					</span>
					<Tabs.List variant="underline">
						<Tabs.Tab
							className="h-auto! flex-col gap-1.5 py-[calc(--spacing(2)-1px)]"
							value="architecture"
						>
							<HouseIcon aria-hidden="true" className="opacity-60" size={16} />
							Architecture
						</Tabs.Tab>
						<Tabs.Tab
							className="h-auto! flex-col gap-1.5 py-[calc(--spacing(2)-1px)]"
							value="model-scope"
						>
							<PanelsTopLeftIcon
								aria-hidden="true"
								className="opacity-60"
								size={16}
							/>
							Model Scope
						</Tabs.Tab>
						<Tabs.Tab
							className="h-auto! flex-col gap-1.5 py-[calc(--spacing(2.5)-1px)]"
							value="research-paper"
						>
							<BoxIcon aria-hidden="true" className="opacity-60" size={16} />
							Research Paper
						</Tabs.Tab>
					</Tabs.List>
				</div>
			</Tabs>
			<Flex.Column className="min-h-0 flex-1" fullHeight fullWidth>
				<Outlet />
			</Flex.Column>
		</Flex.Column>
	);
}
