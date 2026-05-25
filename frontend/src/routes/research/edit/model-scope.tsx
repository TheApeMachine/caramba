import { createFileRoute } from "@tanstack/react-router";
import { useMemo } from "react";
import { usePublishAssistantContext } from "#/components/assistant/use-publish-assistant-context";
import { ModelScopeInspector } from "#/components/modelscope";
import { Flex } from "#/components/ui/flex";

function ResearchEditModelScopePanel() {
	usePublishAssistantContext(
		useMemo(
			() => ({
				key: "current_view",
				label: "Current view",
				value: "Model Scope",
				persistent: true,
			}),
			[],
		),
	);

	return (
		<Flex.Column gap={3} padding={4} className="box-border flex-1" fullHeight>
			<ModelScopeInspector />
		</Flex.Column>
	);
}

export const Route = createFileRoute("/research/edit/model-scope")({
	component: ResearchEditModelScopePanel,
});
