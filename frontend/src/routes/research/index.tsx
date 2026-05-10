import { ClientOnly, createFileRoute } from "@tanstack/react-router";
import { ResearchProjectsList } from "#/components/research/research-projects-list";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

const ResearchIndex = () => {
	return (
		<ClientOnly fallback={<ResearchIndexPending />}>
			<ResearchProjectsList />
		</ClientOnly>
	);
}

const ResearchIndexPending = () => {
	return (
		<Flex.Center>
			<Typography.Paragraph variant="muted">
				Loading projects…
			</Typography.Paragraph>
		</Flex.Center>
	);
}

export const Route = createFileRoute("/research/")({
	component: ResearchIndex,
});