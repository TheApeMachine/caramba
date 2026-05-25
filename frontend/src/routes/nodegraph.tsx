import { ClientOnly, createFileRoute } from "@tanstack/react-router";
import { FlumeEditor } from "#/components/flume/flume-editor";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

const NodeGraphRoute = () => (
	<Flex.Column
		gap={3}
		padding={4}
		className="box-border min-h-0 min-h-screen flex-1"
		fullHeight
		fullWidth
	>
		<Flex.Column gap={1} className="shrink-0">
			<h1 className="font-semibold text-foreground text-lg">Architecture</h1>
			<Typography.Paragraph variant="muted">
				Build and manage research architectures with the graph editor.
			</Typography.Paragraph>
		</Flex.Column>
		<Flex.Column className="min-h-0 flex-1" fullHeight fullWidth>
			<ClientOnly
				fallback={
					<Flex.Center className="min-h-[75vh] flex-1">
						<Typography.Paragraph variant="muted">
							Loading graph editor…
						</Typography.Paragraph>
					</Flex.Center>
				}
			>
				<FlumeEditor />
			</ClientOnly>
		</Flex.Column>
	</Flex.Column>
);

export const Route = createFileRoute("/nodegraph")({
	ssr: false,
	component: NodeGraphRoute,
});
