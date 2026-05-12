import { createFileRoute } from "@tanstack/react-router";
import { ResearchIndex } from "#/routes/research";

const Home = () => {
	return <ResearchIndex />;
};

export const Route = createFileRoute("/")({ component: Home });
