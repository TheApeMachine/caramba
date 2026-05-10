"use client";

import type { ReactFormExtendedApi } from "@tanstack/react-form";
import { useForm } from "@tanstack/react-form";
import type { PaperMetadata } from "#/components/latex/model/types";
import { Field } from "#/components/ui/field";
import { Flex } from "#/components/ui/flex";
import { Input } from "#/components/ui/input";
import { Textarea } from "#/components/ui/textarea";

export type PaperMetadataFormApi = ReactFormExtendedApi<
	PaperMetadata,
	undefined,
	undefined,
	undefined,
	undefined,
	undefined,
	undefined,
	undefined,
	undefined,
	undefined,
	undefined,
	undefined
>;

/** Typed so `PaperMetadataFormApi` matches (inference alone widens validators / submit meta). */
export function usePaperMetadataForm(): PaperMetadataFormApi {
	return useForm<
		PaperMetadata,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined,
		undefined
	>({
		defaultValues: {
			title: "",
			authors: "",
			keywords: "",
			abstract: "",
		},
	});
}

export function MetadataTab({ form }: { form: PaperMetadataFormApi }) {
	return (
		<form
			onSubmit={(e) => {
				e.preventDefault();
			}}
		>
			<Flex.Column gap={4} padding={3}>
				<form.Field name="title">
					{(field) => (
						<Field>
							<Field.Label htmlFor={field.name}>Title</Field.Label>
							<Input
								id={field.name}
								name={field.name}
								onBlur={field.handleBlur}
								onChange={(e) => field.handleChange(e.target.value)}
								placeholder="Paper title"
								value={field.state.value}
							/>
						</Field>
					)}
				</form.Field>
				<form.Field name="authors">
					{(field) => (
						<Field>
							<Field.Label htmlFor={field.name}>Authors</Field.Label>
							<Textarea
								id={field.name}
								name={field.name}
								onBlur={field.handleBlur}
								onChange={(e) => field.handleChange(e.target.value)}
								placeholder="One author per line"
								value={field.state.value}
							/>
						</Field>
					)}
				</form.Field>
				<form.Field name="keywords">
					{(field) => (
						<Field>
							<Field.Label htmlFor={field.name}>Keywords</Field.Label>
							<Input
								id={field.name}
								name={field.name}
								onBlur={field.handleBlur}
								onChange={(e) => field.handleChange(e.target.value)}
								placeholder="Comma-separated"
								value={field.state.value}
							/>
						</Field>
					)}
				</form.Field>
				<form.Field name="abstract">
					{(field) => (
						<Field>
							<Field.Label htmlFor={field.name}>Abstract</Field.Label>
							<Textarea
								id={field.name}
								name={field.name}
								onBlur={field.handleBlur}
								onChange={(e) => field.handleChange(e.target.value)}
								placeholder="Short abstract"
								rows={6}
								value={field.state.value}
							/>
						</Field>
					)}
				</form.Field>
			</Flex.Column>
		</form>
	);
}
