import {
	ClientOnly,
	createFileRoute,
	getRouteApi,
	Link,
} from "@tanstack/react-router";
import { useMemo } from "react";
import { usePublishAssistantContext } from "#/components/assistant/use-publish-assistant-context";
import { FlumeEditor } from "#/components/flume/flume-editor";
import { Button } from "#/components/ui/button";
import { Flex } from "#/components/ui/flex";

const researchEditRouteApi = getRouteApi("/research/edit");

const ResearchEditArchitecturePanel = () => {
	const search = researchEditRouteApi.useSearch();

	usePublishAssistantContext(
		useMemo(
			() => ({
				key: "current_view",
				label: "Current view",
				value: "Research graph",
				persistent: true,
			}),
			[],
		),
	);

	usePublishAssistantContext(
		useMemo(
			() =>
				search.projectId
					? {
							key: "project_id",
							label: "Project ID",
							value: search.projectId,
							persistent: true,
						}
					: null,
			[search.projectId],
		),
	);

	return (
		<Flex.Column
			gap={3}
			padding={4}
			className="box-border min-h-0 flex-1"
			fullHeight
			fullWidth
		>
			<Flex.Row gap={2} className="shrink-0 items-center justify-between gap-2">
				<Flex.Column gap={1}>
					<h1 className="font-semibold text-foreground text-lg">
						Research graph
					</h1>
					{search.projectId ? (
						<p className="font-mono text-muted-foreground text-xs">
							{search.projectId}
						</p>
					) : (
						<p className="text-muted-foreground text-sm">
							No project id in URL — graph is still editable locally.
						</p>
					)}
				</Flex.Column>
				<Button render={<Link to="/research" />} size="sm" variant="outline">
					Back to projects
				</Button>
			</Flex.Row>
			<Flex.Column className="min-h-0 flex-1" fullHeight fullWidth>
				<ClientOnly
					fallback={
						<Flex.Center className="min-h-48 flex-1">
							<p className="text-muted-foreground text-sm">Loading graph…</p>
						</Flex.Center>
					}
				>
					<FlumeEditor projectId={search.projectId} />
				</ClientOnly>
			</Flex.Column>
		</Flex.Column>
	);
};

export const Route = createFileRoute("/research/edit/")({
	component: ResearchEditArchitecturePanel,
});
