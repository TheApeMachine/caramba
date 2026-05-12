import { z } from "zod";

/** Row shape for the `research_projects` table synced via Electric. */
export const researchProjectRowSchema = z.object({
	id: z.uuid(),
	name: z.string().min(1),
	description: z.string(),
	organization_slug: z.string().default(""),
	project_slug: z.string().min(1).nullable().optional(),
	created_at: z.coerce.date(),
	updated_at: z.coerce.date(),
});

export type ResearchProjectRow = z.infer<typeof researchProjectRowSchema>;
