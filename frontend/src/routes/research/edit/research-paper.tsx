import {
	ClientOnly,
	createFileRoute,
	getRouteApi,
	useNavigate,
} from "@tanstack/react-router";
import { useCallback } from "react";
import { PaperEditorApp } from "#/components/latex/component";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

const researchEditRouteApi = getRouteApi("/research/edit");

function ResearchEditPaperPanel() {
	const navigate = useNavigate();
	const search = researchEditRouteApi.useSearch();

	const handlePaperBootstrapped = useCallback(
		(paperId: string) => {
			void navigate({
				replace: true,
				search: (previous) => ({
					...previous,
					paperId,
				}),
			});
		},
		[navigate],
	);

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
				<PaperEditorApp
					onPaperBootstrapped={handlePaperBootstrapped}
					paperId={search.paperId}
					projectId={search.projectId}
				/>
			</Flex.Column>
		</ClientOnly>
	);
}

export const Route = createFileRoute("/research/edit/research-paper")({
	staticData: { pageContentWidth: "contained" },
	component: ResearchEditPaperPanel,
});
