import { useUser } from "@clerk/tanstack-react-start";
import { createFileRoute } from "@tanstack/react-router";
import { useTranslation } from "react-i18next";
import { ResearchDashboard } from "#/components/research/research-dashboard";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

const greetingForHour = (hour: number, t: (key: string) => string) => {
	if (hour < 12) {
		return t("research.greetingMorning");
	}

	if (hour < 18) {
		return t("research.greetingAfternoon");
	}

	return t("research.greetingEvening");
};

const ResearchHomeHeader = () => {
	const { user, isLoaded } = useUser();
	const { t } = useTranslation();
	const hour = new Date().getHours();
	const greeting = greetingForHour(hour, t);
	const name =
		user?.firstName ?? user?.username ?? t("research.researcherFallback");

	return (
		<Flex.Column gap={1} className="shrink-0">
			<h1 className="font-semibold text-2xl text-foreground tracking-tight">
				{isLoaded ? `${greeting}, ${name}` : t("research.workspaceLoading")}
			</h1>
			<Typography.Paragraph variant="muted">
				{t("research.subtitle")}
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
