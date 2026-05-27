/*
Jazz 2.0 row-level permissions — replaces the Electric proxy's
`where organization_slug = $1` scoping and the *_members RLS checks.

API verified against jazz-tools@2.0.0-alpha.50 (dist/permissions/index.d.ts):
  s.definePermissions(app, (ctx) => void)
  ctx.policy.<table>.allowRead|allowInsert|allowUpdate|allowDelete
  .where(input | (row) => condition) | .always() | .never()
  ctx.policy.<table>.exists.where({ ... })        -> membership / "shares" check
  ctx.session.user_id                              -> authenticated user (JWT sub)
  ctx.session.where({ "claims.x": v })             -> assert a JWT claim
  ctx.allowedTo.read("<refColumn>")                -> inherit access via a ref
  ctx.anyOf([...]) / ctx.allOf([...]) / ctx.isCreator

Model: access is membership-driven. A user can touch a project's data iff they
are a row in projectMembers for that project; team data iff in teamMemberships.
This is the documented exists/relation pattern and avoids depending on a custom
Clerk JWT template on day one. (To scope by org claim instead, add an org claim
to the Clerk JWT template and use ctx.session.where({ "claims.org_slug": ... }).)
*/

import { schema as s } from "jazz-tools";
import { app } from "./schema";

export default s.definePermissions(app, ({ policy, session, allowedTo }) => {
	// --- Teams: members read; owners update; anyone signed-in can create ---
	policy.teams.allowInsert.always();
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

	// Membership rows: a user sees/creates their own. Inviting others needs an
	// owner-gated flow — left as a follow-up.
	policy.teamMemberships.allowRead.where({ user_id: session.user_id });
	policy.teamMemberships.allowInsert.where({ user_id: session.user_id });
	policy.projectMembers.allowRead.where({ user_id: session.user_id });
	policy.projectMembers.allowInsert.where({ user_id: session.user_id });

	// --- Projects: members read; owners update; anyone signed-in can create ---
	policy.projects.allowInsert.always(); // app adds the creator as an owner member
	policy.projects.allowRead.where((project) =>
		policy.projectMembers.exists.where({
			project: project.id,
			user_id: session.user_id,
		}),
	);
	policy.projects.allowUpdate.where((project) =>
		policy.projectMembers.exists.where({
			project: project.id,
			user_id: session.user_id,
			role: "owner",
		}),
	);

	// --- Kanban cards: scoped to project membership ---
	policy.kanbanCards.allowRead.where((card) =>
		policy.projectMembers.exists.where({
			project: card.project,
			user_id: session.user_id,
		}),
	);
	policy.kanbanCards.allowInsert.where((card) =>
		policy.projectMembers.exists.where({
			project: card.project,
			user_id: session.user_id,
		}),
	);
	policy.kanbanCards.allowUpdate.where((card) =>
		policy.projectMembers.exists.where({
			project: card.project,
			user_id: session.user_id,
		}),
	);
	policy.kanbanCards.allowDelete.where((card) =>
		policy.projectMembers.exists.where({
			project: card.project,
			user_id: session.user_id,
		}),
	);

	// Subtasks inherit access from their parent card.
	policy.kanbanSubtasks.allowRead.where(allowedTo.read("card"));
	policy.kanbanSubtasks.allowInsert.where(allowedTo.read("card"));
	policy.kanbanSubtasks.allowUpdate.where(allowedTo.read("card"));
	policy.kanbanSubtasks.allowDelete.where(allowedTo.read("card"));

	// --- Papers: scoped to project membership; blocks inherit from paper ---
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
});
