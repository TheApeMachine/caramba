"use client";

import {
	type Context,
	type InitialQueryBuilder,
	type QueryBuilder,
	useLiveQuery,
} from "@tanstack/react-db";
import { ClientOnly } from "@tanstack/react-router";
import type { ReactNode } from "react";
import { Loadable } from "#/components/ui/loadable";
import { LoadablePending } from "#/components/ui/loadable/pending";

type LiveQueryFn = (query: InitialQueryBuilder) => QueryBuilder<Context>;

const defaultIsEmpty = <TData,>(data: TData): boolean => {
	if (data == null) {
		return true;
	}

	if (Array.isArray(data)) {
		return data.length === 0;
	}

	return false;
};

interface ComponentProps<TData> {
	name: string;
	query: LiveQueryFn;
	children: (data: TData) => ReactNode;
	isEmpty?: (data: TData) => boolean;
	errorMessage?: string;
	pending?: ReactNode;
	error?: ReactNode;
	empty?: ReactNode;
}

const ComponentBody = <TData,>({
	name,
	query,
	children,
	isEmpty = defaultIsEmpty,
	errorMessage,
	pending,
	error,
	empty,
}: ComponentProps<TData>) => {
	const { data, isLoading, isError } = useLiveQuery(query);
	const resolved = data as TData;
	const empty_ = !isLoading && !isError && isEmpty(resolved);

	return (
		<Loadable
			name={name}
			isLoading={isLoading}
			isError={isError}
			isEmpty={empty_}
			errorMessage={errorMessage}
			pending={pending}
			error={error}
			empty={empty}
		>
			{children(resolved)}
		</Loadable>
	);
};

/*
Component wraps a single useLiveQuery in the shared Loadable chrome.
Multi-query callers consume Loadable directly with their own
useLiveQuery calls and aggregate the flags.
*/
export const Component = <TData,>(props: ComponentProps<TData>) => (
	<ClientOnly fallback={<LoadablePending name={props.name} />}>
		<ComponentBody {...props} />
	</ClientOnly>
);
