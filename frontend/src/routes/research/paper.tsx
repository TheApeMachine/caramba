import { ClientOnly, createFileRoute } from "@tanstack/react-router";
import { PaperEditorApp } from "#/components/latex/component";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

export const Route = createFileRoute("/research/paper")({
	component: ResearchPaperRoute,
});

function ResearchPaperRoute() {
	return (
		<ClientOnly
			fallback={
				<Flex.Center className="min-h-[50dvh] flex-1 p-4">
					<Typography.Paragraph variant="muted">
						Loading editor…
					</Typography.Paragraph>
				</Flex.Center>
			}
		>
			<PaperEditorApp />
		</ClientOnly>
	);
}
