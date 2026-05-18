"use client";

import {
	type Context,
	type InitialQueryBuilder,
	type QueryBuilder,
	useLiveQuery,
} from "@tanstack/react-db";
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
	pending?: ReactNode;
	error?: ReactNode;
	empty?: ReactNode;
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
	<Flex.Center className="min-h-24 flex-1 flex-col gap-2 p-6">
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
	<Alert variant="error">
		<CircleAlertIcon />
		<AlertTitle>Error loading {name}</AlertTitle>
		<AlertDescription>
			{errorMessage ??
				`An error occurred while loading ${name}. Please try again later.`}
		</AlertDescription>
		<AlertAction>
			<Button size="xs">Report Issue</Button>
		</AlertAction>
	</Alert>
);

const ComponentEmpty = ({ name }: { name: string }) => (
	<Empty>
		<Empty.Header>
			<Empty.Media variant="icon">
				<RouteIcon />
			</Empty.Media>
			<Empty.Title>No {name}</Empty.Title>
			<Empty.Description>Create a {name} to get started.</Empty.Description>
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
);

/**
 * Component wraps live-query-driven UI with shared loading, error, and empty states.
 */
export const Component = <TData,>({
	children,
	name,
	query,
	errorMessage,
	isEmpty = defaultIsEmpty,
	pending,
	error,
	empty,
}: ComponentProps<TData>) => {
	const { data, isLoading, isError } = useLiveQuery(query);

	if (isLoading) {
		return pending ?? <ComponentPending name={name} />;
	}

	if (isError) {
		return error ?? <ComponentError name={name} errorMessage={errorMessage} />;
	}

	const resolvedData = data as TData;

	if (isEmpty(resolvedData)) {
		return empty ?? <ComponentEmpty name={name} />;
	}

	return children(resolvedData);
};
