import { createFileRoute } from "@tanstack/react-router";
import {
	type ResearchProjectRow,
	researchProjectCollection,
} from "#/collections/research_project";
import { Component } from "#/components/component";
import { TodoComponent } from "#/components/todo/component";
import { DataTable } from "#/components/ui/datatable/component";
import { Grid } from "#/components/ui/grid";

const ResearchIndex = () => (
	<Component<ResearchProjectRow[]>
		name="research projects"
		query={(query) => query.from({ project: researchProjectCollection })}
	>
		{(projects) => {
			return (
				<Grid.Bento cols={6}>
					<Grid.Span cols={3} rows={1}>
						<DataTable
							data={projects}
							columns={["name", "description", "created_at"]}
						/>
					</Grid.Span>
					<Grid.Span cols={2} rows={1}>
						<TodoComponent />
					</Grid.Span>
					<Grid.Span cols={2} rows={1}>
						<TodoComponent />
					</Grid.Span>
					<Grid.Span cols={2} rows={1}>
						<TodoComponent />
					</Grid.Span>
					<Grid.Span cols={2} rows={1}>
						<TodoComponent />
					</Grid.Span>
				</Grid.Bento>
			);
		}}
	</Component>
);

export const Route = createFileRoute("/research/")({
	ssr: false,
	component: ResearchIndex,
});
