import { ClientOnly, createFileRoute } from "@tanstack/react-router";
import { ResearchProjectsList } from "#/components/research/research-projects-list";
import { Flex } from "#/components/ui/flex";
import { Typography } from "#/components/ui/typography";

const Home = () => {
	return (
		<ClientOnly fallback={<HomePending />}>
			<ResearchProjectsList />
		</ClientOnly>
	);
};

/* 
SSR-safe placeholder
`useLiveQuery` uses `useSyncExternalStore` without getServerSnapshot. 
*/
const HomePending = () => {
	return (
		<Flex.Center>
			<Typography.Paragraph variant="muted">
				Loading projects…
			</Typography.Paragraph>
		</Flex.Center>
	);
};

export const Route = createFileRoute("/")({ component: Home });
