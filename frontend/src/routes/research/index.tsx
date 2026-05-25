import { useUser } from "@clerk/tanstack-react-start";
import { createFileRoute } from "@tanstack/react-router";
import { ResearchDashboard } from "#/components/research/research-dashboard";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

const greetingForHour = (hour: number) => {
	if (hour < 12) {
		return "Good morning";
	}

	if (hour < 18) {
		return "Good afternoon";
	}

	return "Good evening";
};

const ResearchHomeHeader = () => {
	const { user, isLoaded } = useUser();
	const hour = new Date().getHours();
	const greeting = greetingForHour(hour);
	const name = user?.firstName ?? user?.username ?? "Researcher";

	return (
		<Flex.Column gap={1} className="shrink-0">
			<h1 className="font-semibold text-2xl text-foreground tracking-tight">
				{isLoaded ? `${greeting}, ${name}` : "Research workspace"}
			</h1>
			<Typography.Paragraph variant="muted">
				Your projects, tasks, and recent activity in one place.
			</Typography.Paragraph>
		</Flex.Column>
	);
};

const ResearchIndex = () => (
	<div className="flex h-full min-h-0 w-full flex-1 flex-col gap-6 p-4 sm:p-6">
		<ResearchHomeHeader />
		<div className="min-h-0 flex-1">
			<ResearchDashboard />
		</div>
	</div>
);

export const Route = createFileRoute("/research/")({
	ssr: false,
	component: ResearchIndex,
});
