import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useCallback } from "react";
import { z } from "zod";
import { PaperEditorApp } from "#/components/latex/component";

const paperSearchSchema = z.object({
	paperId: z.uuid().optional(),
	projectId: z.uuid().optional(),
});

function parsePaperEditSearch(
	raw: Record<string, unknown>,
): z.infer<typeof paperSearchSchema> {
	const parsed = paperSearchSchema.safeParse(raw);

	return parsed.success
		? parsed.data
		: { paperId: undefined, projectId: undefined };
}

const RouteComponent = () => {
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
		<PaperEditorApp
			onPaperBootstrapped={handlePaperBootstrapped}
			paperId={search.paperId}
			projectId={search.projectId}
		/>
	);
};

export const Route = createFileRoute("/paper/edit")({
	validateSearch: parsePaperEditSearch,
	staticData: { pageContentWidth: "contained" },
	component: RouteComponent,
});
