import { useForm } from "@tanstack/react-form";
import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "#/components/ui/alert";
import { Button } from "#/components/ui/button";
import { Field } from "#/components/ui/field";
import { Flex } from "#/components/ui/flex";
import { Input } from "#/components/ui/input";
import { Textarea } from "#/components/ui/textarea";
import { Typography } from "#/components/ui/typography";
import { submitFeatureRequest } from "#/server/feature-request";

export const Route = createFileRoute("/request-feature")({
	component: RequestFeatureRoute,
});

function RequestFeatureRoute() {
	const [submissionError, setSubmissionError] = useState<string | null>(null);
	const [submissionSuccess, setSubmissionSuccess] = useState<string | null>(
		null,
	);

	const form = useForm({
		defaultValues: {
			title: "",
			description: "",
			contact_email: "",
		},
		onSubmit: async ({ value }) => {
			setSubmissionError(null);
			setSubmissionSuccess(null);

			try {
				await submitFeatureRequest({
					data: {
						title: value.title.trim(),
						description: value.description.trim(),
						contact_email:
							value.contact_email.trim().length > 0
								? value.contact_email.trim()
								: undefined,
					},
				});

				setSubmissionSuccess(
					"Thanks — your request was added to the Requests backlog.",
				);
			} catch (error) {
				setSubmissionError(
					error instanceof Error ? error.message : String(error),
				);
			}
		},
	});

	return (
		<Flex.Center padding={4} className="min-h-full">
			<Flex.Column gap={6} className="w-full max-w-lg">
				<Flex.Column gap={1}>
					<h1 className="font-semibold text-foreground text-lg">
						Request a feature
					</h1>
					<Typography.Paragraph variant="muted">
						Requests become Backlog cards on the Requests project for the
						configured organization (seeded as{" "}
						<code className="text-foreground">caramba / requests</code> by
						default).
					</Typography.Paragraph>
				</Flex.Column>

				{submissionSuccess !== null ? (
					<Alert variant="success">
						<AlertTitle>Submitted</AlertTitle>
						<AlertDescription>{submissionSuccess}</AlertDescription>
					</Alert>
				) : null}

				{submissionError !== null ? (
					<Alert variant="error">
						<AlertTitle>Could not submit</AlertTitle>
						<AlertDescription>{submissionError}</AlertDescription>
					</Alert>
				) : null}

				<form
					className="flex flex-col gap-4"
					onSubmit={(event) => {
						event.preventDefault();
						event.stopPropagation();
						void form.handleSubmit();
					}}
				>
					<form.Field
						name="title"
						validators={{
							onChange: ({ value }) =>
								value.trim().length < 3 ? "Title is too short" : undefined,
						}}
					>
						{(field) => (
							<Field
								data-invalid={
									field.state.meta.isTouched && !field.state.meta.isValid
								}
							>
								<Field.Label htmlFor={field.name}>Title</Field.Label>
								<Input
									aria-invalid={
										field.state.meta.isTouched && !field.state.meta.isValid
									}
									id={field.name}
									name={field.name}
									onBlur={field.handleBlur}
									onChange={(event) => field.handleChange(event.target.value)}
									value={field.state.value}
								/>
								{field.state.meta.isTouched &&
								field.state.meta.errors.length ? (
									<Field.Error>
										{field.state.meta.errors.join(", ")}
									</Field.Error>
								) : null}
							</Field>
						)}
					</form.Field>

					<form.Field
						name="description"
						validators={{
							onChange: ({ value }) =>
								value.trim().length < 1 ? "Description is required" : undefined,
						}}
					>
						{(field) => (
							<Field
								data-invalid={
									field.state.meta.isTouched && !field.state.meta.isValid
								}
							>
								<Field.Label htmlFor={field.name}>Description</Field.Label>
								<Textarea
									aria-invalid={
										field.state.meta.isTouched && !field.state.meta.isValid
									}
									id={field.name}
									name={field.name}
									onBlur={field.handleBlur}
									onChange={(event) => field.handleChange(event.target.value)}
									value={field.state.value}
								/>
								{field.state.meta.isTouched &&
								field.state.meta.errors.length ? (
									<Field.Error>
										{field.state.meta.errors.join(", ")}
									</Field.Error>
								) : null}
							</Field>
						)}
					</form.Field>

					<form.Field name="contact_email">
						{(field) => (
							<Field>
								<Field.Label htmlFor={field.name}>
									Contact email (optional)
								</Field.Label>
								<Input
									autoComplete="email"
									id={field.name}
									inputMode="email"
									name={field.name}
									onBlur={field.handleBlur}
									onChange={(event) => field.handleChange(event.target.value)}
									placeholder="you@example.com"
									type="email"
									value={field.state.value}
								/>
							</Field>
						)}
					</form.Field>

					<form.Subscribe
						selector={(state) => [state.canSubmit, state.isSubmitting] as const}
					>
						{([canSubmit, isSubmitting]) => (
							<Flex.Row className="justify-end" gap={2}>
								<Button render={<Link to="/kanban" />} variant="outline">
									Kanban hub
								</Button>
								<Button disabled={!canSubmit || isSubmitting} type="submit">
									{isSubmitting ? "Sending…" : "Submit request"}
								</Button>
							</Flex.Row>
						)}
					</form.Subscribe>
				</form>
			</Flex.Column>
		</Flex.Center>
	);
}
