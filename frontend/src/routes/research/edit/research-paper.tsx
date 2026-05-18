import {
	ClientOnly,
	createFileRoute,
	getRouteApi,
	useNavigate,
} from "@tanstack/react-router";
import { useCallback } from "react";
import { PaperEditorApp } from "#/components/latex/component";
import { ProjectPaperSwitcher } from "#/components/research/project-paper-switcher";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

const researchEditRouteApi = getRouteApi("/research/edit");

function ResearchEditPaperPanel() {
	const navigate = useNavigate();
	const search = researchEditRouteApi.useSearch();

	const handleSelectPaperId = useCallback(
		(paperId: string) => {
			void navigate({
				to: "/research/edit/research-paper",
				replace: true,
				search: {
					projectId: search.projectId,
					paperId,
				},
			});
		},
		[navigate, search.projectId],
	);

	const handlePaperBootstrapped = useCallback(
		(paperId: string) => {
			handleSelectPaperId(paperId);
		},
		[handleSelectPaperId],
	);

	if (!search.projectId) {
		return (
			<Flex.Center className="min-h-[50dvh] flex-1 p-4">
				<Typography.Paragraph variant="muted">
					Open a research project to edit its papers.
				</Typography.Paragraph>
			</Flex.Center>
		);
	}

	const showEditor = Boolean(search.paperId);

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
				<ProjectPaperSwitcher
					projectId={search.projectId}
					selectedPaperId={search.paperId}
					onSelectPaperId={handleSelectPaperId}
				/>

				{showEditor ? (
					<PaperEditorApp
						onPaperBootstrapped={handlePaperBootstrapped}
						paperId={search.paperId}
						projectId={search.projectId}
					/>
				) : (
					<Flex.Center className="min-h-[40dvh] flex-1 rounded-xl border border-dashed bg-card/30 p-6">
						<Typography.Paragraph
							className="max-w-md text-center"
							variant="muted"
						>
							Select a paper above, or create a new one. Each paper stays linked
							to this project so you can manage several manuscripts from one
							effort.
						</Typography.Paragraph>
					</Flex.Center>
				)}
			</Flex.Column>
		</ClientOnly>
	);
}

export const Route = createFileRoute("/research/edit/research-paper")({
	staticData: { pageContentWidth: "contained" },
	component: ResearchEditPaperPanel,
});
