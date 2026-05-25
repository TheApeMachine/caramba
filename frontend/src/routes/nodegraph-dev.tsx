import { ClientOnly, createFileRoute } from "@tanstack/react-router";
import { FlumeEditor } from "#/components/flume/flume-editor";
import { Flex } from "#/components/ui/flex";

const NodeGraphDevRoute = () => (
	<Flex.Column className="box-border min-h-screen flex-1" fullHeight fullWidth>
		<ClientOnly
			fallback={
				<Flex.Center className="min-h-screen flex-1">
					<p className="text-muted-foreground text-sm">Loading graph editor…</p>
				</Flex.Center>
			}
		>
			<FlumeEditor />
		</ClientOnly>
	</Flex.Column>
);

export const Route = createFileRoute("/nodegraph-dev")({
	ssr: false,
	component: NodeGraphDevRoute,
});
