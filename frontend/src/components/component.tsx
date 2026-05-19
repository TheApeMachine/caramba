"use client";

import {
	type Context,
	type InitialQueryBuilder,
	type QueryBuilder,
	useLiveQuery,
} from "@tanstack/react-db";
import { ClientOnly } from "@tanstack/react-router";
import { BookIcon, CircleAlertIcon, RouteIcon } from "lucide-react";
import type { ReactNode } from "react";
import {
	Alert,
	AlertAction,
	AlertDescription,
	AlertTitle,
} from "#/components/ui/alert";
import { Button } from "#/components/ui/button";
import { Empty } from "#/components/ui/empty";
import { Flex } from "#/components/ui/flex";
import { Spinner } from "#/components/ui/spinner";
import { Typography } from "#/components/ui/typography";

type LiveQueryFn = (query: InitialQueryBuilder) => QueryBuilder<Context>;

interface ComponentProps<TData> {
	name: string;
	query: LiveQueryFn;
	children: (data: TData) => ReactNode;
	errorMessage?: string;
	isEmpty?: (data: TData) => boolean;
}

const defaultIsEmpty = <TData,>(data: TData): boolean => {
	if (data == null) {
		return true;
	}

	if (Array.isArray(data)) {
		return data.length === 0;
	}

	return false;
};

const ComponentPending = ({ name }: { name: string }) => (
	<Flex.Center className="h-full min-h-0 flex-1 flex-col gap-2 p-4">
		<Spinner className="text-muted-foreground" />
		<Typography.Paragraph variant="muted">Loading {name}…</Typography.Paragraph>
	</Flex.Center>
);

const ComponentError = ({
	name,
	errorMessage,
}: {
	name: string;
	errorMessage?: string;
}) => (
	<Flex.Center className="h-full min-h-0 flex-1 p-4 md:p-8">
		<Alert className="max-w-lg" variant="error">
			<CircleAlertIcon />
			<AlertTitle>Error loading {name}</AlertTitle>
			<AlertDescription>
				{errorMessage ??
					`Could not load ${name}. Check your connection and try again.`}
			</AlertDescription>
			<AlertAction>
				<Button size="xs">Report issue</Button>
			</AlertAction>
		</Alert>
	</Flex.Center>
);

const ComponentEmpty = ({ name }: { name: string }) => (
	<div className="flex h-full min-h-0 w-full flex-1 p-4 md:p-8">
		<Empty className="mx-auto h-full min-h-[min(32rem,100%)] w-full max-w-xl flex-1 rounded-2xl border border-dashed bg-card/30">
			<Empty.Header>
				<Empty.Media variant="icon">
					<RouteIcon className="size-5" />
				</Empty.Media>
				<Empty.Title>No {name} yet</Empty.Title>
				<Empty.Description>
					Create a {name} to get started, or view the docs to learn more.
				</Empty.Description>
			</Empty.Header>
			<Empty.Content>
				<Flex.Row gap={2}>
					<Button size="sm">Create {name}</Button>
					<Button size="sm" variant="outline">
						<BookIcon />
						View docs
					</Button>
				</Flex.Row>
			</Empty.Content>
		</Empty>
	</div>
);

const ComponentBody = <TData,>({
	children,
	name,
	query,
	errorMessage,
	isEmpty = defaultIsEmpty,
}: ComponentProps<TData>) => {
	const { data, isLoading, isError } = useLiveQuery(query);

	if (isLoading) {
		return <ComponentPending name={name} />;
	}

	if (isError) {
		return <ComponentError name={name} errorMessage={errorMessage} />;
	}

	const resolvedData = data as TData;

	if (isEmpty(resolvedData)) {
		return <ComponentEmpty name={name} />;
	}

	return children(resolvedData);
};

/**
 * Component wraps live-query-driven UI with shared loading, error, and empty states.
 */
export const Component = <TData,>(props: ComponentProps<TData>) => (
	<ClientOnly fallback={<ComponentPending name={props.name} />}>
		<ComponentBody {...props} />
	</ClientOnly>
);
