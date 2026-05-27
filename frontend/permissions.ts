/*
Jazz 2.0 row-level permissions — replaces the Electric shape proxy's
`where organization_slug = $1` scoping and the *_members RLS checks.

Confirmed DSL (from Jazz 2.0 docs):
  - permissions function receives (policy, session)
  - policy.<table>.allowRead | allowInsert | allowUpdate | allowDelete
  - .where({ column: session.user_id })  .always()  .where({})
  - anyOf([...]) / allOf([...]) / { not: ... }
  - session.user_id                         -> authenticated user (JWT sub)
  - session.where({ "claims.<x>": value })  -> assert a JWT claim value
  - policy.<table>.exists.where({ ... })     -> membership / "shares" table check
  - relationship helpers: allowedTo.read("<ref>")

VERIFY against the installed jazz-tools@alpha (inferred, not yet runtime-checked):
  1. The exact Clerk JWT claim key for org/role. Clerk's default session token does
     NOT include org claims unless you add them via a JWT template. Decide the claim
     names here (e.g. claims.org_slug, claims.org_role) and mirror them in the Clerk
     JWT template. Until then, membership-table checks below are the safe path.
  2. Whether `exists.where` compares a ref column by id as written here.
  3. Whether `allowedTo.read("project")` chains across the kanbanCards->projects ref.

Model: access is membership-driven. You can read/write a project's data iff you
are a row in projectMembers for that project; team data iff in teamMemberships.
This is the documented "shares table" pattern and avoids depending on a custom
Clerk JWT template on day one.
*/

import type { Permissions } from "jazz-tools";
import { app } from "./schema";

// `allowedTo` is taken as a third parameter here (relationship-inheritance helper
// seen in the docs as `allowedTo.read("<ref>")`). VERIFY its provenance against the
// installed jazz-tools@alpha — it may instead be imported or hung off `policy`.
export default ((policy, session, allowedTo) => {
	// --- Teams: visible to members; mutations by team owners ---
	policy.teams.allowRead.where((team) =>
		policy.teamMemberships.exists.where({
			team: team.id,
			user_id: session.user_id,
		}),
	);
	policy.teams.allowUpdate.where((team) =>
		policy.teamMemberships.exists.where({
			team: team.id,
			user_id: session.user_id,
			role: "owner",
		}),
	);

	// A user may read their own membership rows; writes are owner-gated.
	policy.teamMemberships.allowRead.where({ user_id: session.user_id });
	policy.projectMembers.allowRead.where({ user_id: session.user_id });

	// --- Projects: members read; owners mutate ---
	policy.projects.allowRead.where((project) =>
		policy.projectMembers.exists.where({
			project: project.id,
			user_id: session.user_id,
		}),
	);
	policy.projects.allowInsert.where({}); // any authenticated user can create; creator is added as owner member by the app
	policy.projects.allowUpdate.where((project) =>
		policy.projectMembers.exists.where({
			project: project.id,
			user_id: session.user_id,
			role: "owner",
		}),
	);

	// --- Kanban cards/subtasks: scoped to project membership ---
	for (const table of [policy.kanbanCards] as const) {
		table.allowRead.where((card) =>
			policy.projectMembers.exists.where({
				project: card.project,
				user_id: session.user_id,
			}),
		);
		table.allowInsert.where((card) =>
			policy.projectMembers.exists.where({
				project: card.project,
				user_id: session.user_id,
			}),
		);
		table.allowUpdate.where((card) =>
			policy.projectMembers.exists.where({
				project: card.project,
				user_id: session.user_id,
			}),
		);
		table.allowDelete.where((card) =>
			policy.projectMembers.exists.where({
				project: card.project,
				user_id: session.user_id,
			}),
		);
	}

	// Subtasks inherit access from their parent card's project.
	policy.kanbanSubtasks.allowRead.where(allowedTo.read("card"));
	policy.kanbanSubtasks.allowInsert.where(allowedTo.read("card"));
	policy.kanbanSubtasks.allowUpdate.where(allowedTo.read("card"));
	policy.kanbanSubtasks.allowDelete.where(allowedTo.read("card"));

	// --- Papers + blocks: scoped to project membership ---
	policy.papers.allowRead.where((paper) =>
		policy.projectMembers.exists.where({
			project: paper.project,
			user_id: session.user_id,
		}),
	);
	policy.papers.allowInsert.where((paper) =>
		policy.projectMembers.exists.where({
			project: paper.project,
			user_id: session.user_id,
		}),
	);
	policy.papers.allowUpdate.where((paper) =>
		policy.projectMembers.exists.where({
			project: paper.project,
			user_id: session.user_id,
		}),
	);

	policy.paperBlocks.allowRead.where(allowedTo.read("paper"));
	policy.paperBlocks.allowInsert.where(allowedTo.read("paper"));
	policy.paperBlocks.allowUpdate.where(allowedTo.read("paper"));
	policy.paperBlocks.allowDelete.where(allowedTo.read("paper"));
}) satisfies Permissions<typeof app>;
