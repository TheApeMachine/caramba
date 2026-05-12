import { useAuth } from "@clerk/tanstack-react-start";
import { useForm } from "@tanstack/react-form";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "#/components/ui/alert";
import { Button } from "#/components/ui/button";
import { Field } from "#/components/ui/field";
import { Flex } from "#/components/ui/flex";
import { Input } from "#/components/ui/input";
import { Textarea } from "#/components/ui/textarea";
import { researchProjectsCollection } from "#/lib/research-projects-collection";

const RouteComponent = () => {
	const navigate = useNavigate();
	const [persistError, setPersistError] = useState<string | null>(null);
	const { orgSlug } = useAuth();

	const form = useForm({
		defaultValues: {
			name: "",
			description: "",
		},
		onSubmit: async ({ value }) => {
			setPersistError(null);
			try {
				const id = crypto.randomUUID();
				const tx = researchProjectsCollection.insert({
					id,
					name: value.name.trim(),
					description: value.description.trim(),
					organization_slug: orgSlug ?? "",
					created_at: new Date(),
					updated_at: new Date(),
				});
				await tx.isPersisted.promise;
				await navigate({ search: { projectId: id }, to: "/research/edit" });
			} catch (e) {
				setPersistError(e instanceof Error ? e.message : String(e));
			}
		},
	});

	return (
		<Flex.Center padding={4} className="min-h-full">
			<Flex.Column gap={6} className="w-full max-w-md">
				<Flex.Column gap={1}>
					<h1 className="font-semibold text-foreground text-lg">
						New research project
					</h1>
					<p className="text-muted-foreground text-sm">
						Creates a project row via optimistic insert and your Electric write
						path.
					</p>
				</Flex.Column>

				{persistError ? (
					<Alert variant="error">
						<AlertTitle>Could not create project</AlertTitle>
						<AlertDescription>{persistError}</AlertDescription>
					</Alert>
				) : null}

				<form
					className="flex flex-col gap-4"
					onSubmit={(e) => {
						e.preventDefault();
						e.stopPropagation();
						form.handleSubmit();
					}}
				>
					<form.Field
						name="name"
						validators={{
							onChange: ({ value }) =>
								!value?.trim() ? "Name is required" : undefined,
						}}
					>
						{(field) => (
							<Field
								data-invalid={
									field.state.meta.isTouched && !field.state.meta.isValid
								}
							>
								<Field.Label htmlFor={field.name}>Name</Field.Label>
								<Input
									aria-invalid={
										field.state.meta.isTouched && !field.state.meta.isValid
									}
									id={field.name}
									name={field.name}
									onBlur={field.handleBlur}
									onChange={(e) => field.handleChange(e.target.value)}
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

					<form.Field name="description">
						{(field) => (
							<Field>
								<Field.Label htmlFor={field.name}>Description</Field.Label>
								<Textarea
									id={field.name}
									name={field.name}
									onBlur={field.handleBlur}
									onChange={(e) => field.handleChange(e.target.value)}
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
								<Button render={<Link to="/research" />} variant="outline">
									Cancel
								</Button>
								<Button disabled={!canSubmit || isSubmitting} type="submit">
									{isSubmitting ? "Creating…" : "Create project"}
								</Button>
							</Flex.Row>
						)}
					</form.Subscribe>
				</form>
			</Flex.Column>
		</Flex.Center>
	);
};

export const Route = createFileRoute("/research/new")({
	component: RouteComponent,
});
