import { useForm } from "@tanstack/react-form";
import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { Trans, useTranslation } from "react-i18next";
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
	const { t } = useTranslation();
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

				setSubmissionSuccess(t("requestFeature.success"));
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
						{t("requestFeature.title")}
					</h1>
					<Typography.Paragraph variant="muted">
						<Trans
							i18nKey="requestFeature.description"
							components={{
								1: <code className="text-foreground" />,
							}}
						/>
					</Typography.Paragraph>
				</Flex.Column>

				{submissionSuccess !== null ? (
					<Alert variant="success">
						<AlertTitle>{t("requestFeature.submitted")}</AlertTitle>
						<AlertDescription>{submissionSuccess}</AlertDescription>
					</Alert>
				) : null}

				{submissionError !== null ? (
					<Alert variant="error">
						<AlertTitle>{t("requestFeature.submitError")}</AlertTitle>
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
								value.trim().length < 3
									? t("requestFeature.titleTooShort")
									: undefined,
						}}
					>
						{(field) => (
							<Field
								data-invalid={
									field.state.meta.isTouched && !field.state.meta.isValid
								}
							>
								<Field.Label htmlFor={field.name}>
									{t("requestFeature.titleLabel")}
								</Field.Label>
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
								value.trim().length < 1
									? t("requestFeature.descriptionRequired")
									: undefined,
						}}
					>
						{(field) => (
							<Field
								data-invalid={
									field.state.meta.isTouched && !field.state.meta.isValid
								}
							>
								<Field.Label htmlFor={field.name}>
									{t("requestFeature.descriptionLabel")}
								</Field.Label>
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
									{t("requestFeature.contactEmailLabel")}
								</Field.Label>
								<Input
									autoComplete="email"
									id={field.name}
									inputMode="email"
									name={field.name}
									onBlur={field.handleBlur}
									onChange={(event) => field.handleChange(event.target.value)}
									placeholder={t("requestFeature.contactEmailPlaceholder")}
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
									{t("requestFeature.kanbanHub")}
								</Button>
								<Button disabled={!canSubmit || isSubmitting} type="submit">
									{isSubmitting
										? t("requestFeature.sending")
										: t("requestFeature.submit")}
								</Button>
							</Flex.Row>
						)}
					</form.Subscribe>
				</form>
			</Flex.Column>
		</Flex.Center>
	);
}
