import { ClientOnly, createFileRoute, Link } from "@tanstack/react-router";
import { z } from "zod";
import { NodeGraph } from "#/components/nodegraph/component";
import { Button } from "#/components/ui/button";
import { Flex } from "#/components/ui/flex";

const editSearchSchema = z.object({
	projectId: z.uuid().optional(),
});

function parseEditSearch(
	raw: Record<string, unknown>,
): z.infer<typeof editSearchSchema> {
	const parsed = editSearchSchema.safeParse(raw);
	return parsed.success ? parsed.data : { projectId: undefined };
}

export const Route = createFileRoute("/research/edit")({
	validateSearch: parseEditSearch,
	component: ResearchEdit,
});

function ResearchEdit() {
	const { projectId } = Route.useSearch();

	return (
		<Flex.Column gap={3} padding={4} className="box-border flex-1" fullHeight>
			<Flex.Row gap={2} className="shrink-0 items-center justify-between gap-2">
				<Flex.Column gap={1}>
					<h1 className="font-semibold text-foreground text-lg">
						Research graph
					</h1>
					{projectId ? (
						<p className="font-mono text-muted-foreground text-xs">
							{projectId}
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
					<NodeGraph />
				</ClientOnly>
			</Flex.Column>
		</Flex.Column>
	);
}
