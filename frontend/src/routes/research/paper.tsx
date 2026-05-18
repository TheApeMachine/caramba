import {
	ClientOnly,
	createFileRoute,
	useNavigate,
} from "@tanstack/react-router";
import { useCallback } from "react";
import { z } from "zod";
import { PaperEditorApp } from "#/components/latex/component";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

const paperSearchSchema = z.object({
	paperId: z.uuid().optional(),
	projectId: z.uuid().optional(),
});

function parsePaperSearch(
	raw: Record<string, unknown>,
): z.infer<typeof paperSearchSchema> {
	const parsed = paperSearchSchema.safeParse(raw);

	return parsed.success
		? parsed.data
		: { paperId: undefined, projectId: undefined };
}

export const Route = createFileRoute("/research/paper")({
	validateSearch: parsePaperSearch,
	staticData: { pageContentWidth: "contained" },
	component: ResearchPaperRoute,
});

function ResearchPaperRoute() {
	const navigate = useNavigate();
	const search = Route.useSearch();

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
			<PaperEditorApp
				onPaperBootstrapped={handlePaperBootstrapped}
				paperId={search.paperId}
				projectId={search.projectId}
			/>
		</ClientOnly>
	);
}
