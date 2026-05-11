import { createFileRoute } from "@tanstack/react-router";
import { ModelScopeInspector } from "#/components/modelscope";
import { Flex } from "#/components/ui/flex";

function ResearchEditModelScopePanel() {
	return (
		<Flex.Column gap={3} padding={4} className="box-border flex-1" fullHeight>
			<ModelScopeInspector />
		</Flex.Column>
	);
}

export const Route = createFileRoute("/research/edit/model-scope")({
	component: ResearchEditModelScopePanel,
});
