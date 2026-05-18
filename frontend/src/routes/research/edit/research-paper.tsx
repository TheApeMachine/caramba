import { ClientOnly, createFileRoute } from "@tanstack/react-router";
import { PaperEditorApp } from "#/components/latex/component";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

function ResearchEditPaperPanel() {
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
			<Flex.Column gap={3} padding={4} className="box-border" fullHeight>
				<PaperEditorApp />
			</Flex.Column>
		</ClientOnly>
	);
}

export const Route = createFileRoute("/research/edit/research-paper")({
	staticData: { pageContentWidth: "contained" },
	component: ResearchEditPaperPanel,
});
