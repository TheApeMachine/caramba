import { useAuth } from "@clerk/tanstack-react-start";
import { ClientOnly, createFileRoute } from "@tanstack/react-router";
import { CollectionTable } from "#/components/ui/datatable/component";
import { Spinner } from "#/components/ui/spinner";
import { researchProjectsCollection } from "#/lib/research-projects-collection";

function ResearchIndexInner() {
	const { orgSlug } = useAuth();

	if (!orgSlug) {
		return <Spinner />;
	}

	return (
		<CollectionTable
			query={(q) =>
				q
					.from({ project: researchProjectsCollection(orgSlug) })
					.select(({ project }) => ({
						id: project.id,
						name: project.name,
						description: project.description,
						organization_slug: project.organization_slug,
						project_slug: project.project_slug,
						created_at: project.created_at,
					}))
			}
			columns={[
				"id",
				"name",
				"description",
				"organization_slug",
				"project_slug",
				"created_at",
			]}
			selectable
			defaultSortKey="created_at"
			defaultSortDesc
			errorMessage="Could not sync research projects. Check VITE_ELECTRIC_SHAPE_URL."
		/>
	);
}

export const ResearchIndex = () => {
	return (
		<div className="flex flex-col gap-4 p-6">
			<h1 className="font-semibold text-foreground text-lg">
				Research projects
			</h1>
			<ClientOnly fallback={<Spinner />}>
				<ResearchIndexInner />
			</ClientOnly>
		</div>
	);
};

export const Route = createFileRoute("/research/")({
	component: ResearchIndex,
});
