"use client";

import { useLiveQuery } from "@tanstack/react-db";
import { ClientOnly, Link } from "@tanstack/react-router";
import { MicroscopeIcon, PlusIcon } from "lucide-react";
import { researchProjectCollection } from "#/collections/research_project";
import { Dashboard } from "#/components/dashboard";
import { Button } from "#/components/ui/button";
import { Empty } from "#/components/ui/empty";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";
import {
	defaultResearchLayout,
	researchWidgets,
} from "#/routes/research/widgets";

const ResearchHubPending = () => (
	<Flex.Center className="h-full min-h-0 flex-1 p-4">
		<Typography.Paragraph variant="muted">
			Loading research workspace…
		</Typography.Paragraph>
	</Flex.Center>
);

const ResearchHubNoProjects = () => (
	<div className="flex h-full min-h-0 w-full flex-1 p-4 md:p-8">
		<Empty className="mx-auto h-full min-h-[min(32rem,100%)] w-full max-w-xl flex-1 rounded-2xl border border-dashed bg-card/30">
			<Empty.Header>
				<Empty.Media variant="icon">
					<MicroscopeIcon className="size-5" />
				</Empty.Media>
				<Empty.Title>Start your first research project</Empty.Title>
				<Empty.Description>
					Projects bring together your model graph, Kanban board, papers, and
					team. Once you create one, this page becomes your research dashboard.
				</Empty.Description>
			</Empty.Header>
			<Empty.Content>
				<Button
					className="h-12 min-w-[min(100%,16rem)] px-8 text-base"
					render={<Link to="/research/new" />}
					size="lg"
				>
					<PlusIcon className="size-5" />
					Create research project
				</Button>
			</Empty.Content>
		</Empty>
	</div>
);

const ResearchHubDashboard = () => (
	<div className="flex h-full min-h-0 w-full flex-1 p-4">
		<Dashboard
			widgets={researchWidgets}
			initialLayout={defaultResearchLayout}
		/>
	</div>
);

const ResearchHubBody = () => {
	const { data, isLoading, isError } = useLiveQuery((query) =>
		query.from({ project: researchProjectCollection }),
	);

	if (isLoading) {
		return <ResearchHubPending />;
	}

	if (isError) {
		return (
			<Flex.Center className="h-full min-h-0 flex-1 p-4">
				<Typography.Paragraph className="text-destructive text-center">
					Could not load research projects. Check your connection and try again.
				</Typography.Paragraph>
			</Flex.Center>
		);
	}

	const projects = data ?? [];

	if (projects.length === 0) {
		return <ResearchHubNoProjects />;
	}

	return <ResearchHubDashboard />;
};

export const ResearchHub = () => (
	<ClientOnly fallback={<ResearchHubPending />}>
		<ResearchHubBody />
	</ClientOnly>
);
