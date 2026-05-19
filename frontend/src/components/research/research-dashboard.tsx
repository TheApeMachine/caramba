"use client";

import { useMemo } from "react";
import { Dashboard } from "#/components/dashboard";
import {
	type BentoTileSpec,
	packBentoLayout,
} from "#/components/dashboard/grid";
import type { WidgetDescriptor } from "#/components/dashboard/registry";
import { TodoComponent } from "#/components/todo/component";
import { ResearchActivityWidget } from "./activity-widget";
import { ResearchProjectsWidget } from "./projects-widget";
import { QuickActionsWidget } from "./quick-actions-widget";
import { ResearcherProfileWidget } from "./researcher-profile-widget";
import { ResearchStatsWidget } from "./stats-widget";

const researchBentoTiles: BentoTileSpec[] = [
	{
		id: "r-actions",
		kind: "research-actions",
		colSpan: 1,
		rowSpan: 2,
		priority: 100,
	},
	{
		id: "r-stats",
		kind: "research-stats",
		colSpan: 2,
		rowSpan: 1,
		priority: 80,
	},
	{
		id: "r-profile",
		kind: "research-profile",
		colSpan: 1,
		rowSpan: 1,
		priority: 70,
	},
	{
		id: "r-projects",
		kind: "research-projects",
		colSpan: 1,
		rowSpan: 1,
		priority: 60,
	},
	{
		id: "r-todos",
		kind: "research-todos",
		colSpan: 2,
		rowSpan: 1,
		priority: 50,
	},
	{
		id: "r-activity",
		kind: "research-activity",
		colSpan: 1,
		rowSpan: 1,
		priority: 40,
	},
];

const researchWidgets: WidgetDescriptor[] = [
	{
		kind: "research-actions",
		title: "",
		description: "",
		hideHeader: true,
		render: () => <QuickActionsWidget />,
	},
	{
		kind: "research-stats",
		title: "At a glance",
		description: "Project and task counts",
		render: () => <ResearchStatsWidget />,
	},
	{
		kind: "research-projects",
		title: "Projects",
		description: "Your research projects",
		render: () => <ResearchProjectsWidget />,
	},
	{
		kind: "research-activity",
		title: "Recent activity",
		description: "Latest updates across the workspace",
		render: () => <ResearchActivityWidget />,
	},
	{
		kind: "research-profile",
		title: "",
		description: "",
		hideHeader: true,
		render: () => <ResearcherProfileWidget />,
	},
	{
		kind: "research-todos",
		title: "My todos",
		description: "Kanban cards assigned to you",
		render: () => <TodoComponent />,
	},
];

export const ResearchDashboard = () => {
	const initialLayout = useMemo(() => packBentoLayout(researchBentoTiles), []);

	return (
		<Dashboard
			widgets={researchWidgets}
			initialLayout={initialLayout}
			layoutEditable={false}
		/>
	);
};
