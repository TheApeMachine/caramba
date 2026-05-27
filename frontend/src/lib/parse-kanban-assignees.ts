/*
parseKanbanAssignees normalizes a card's assignees into a string[].

Jazz stores assignees as a native JSON array (schema: assignees s.json()), so the
value arrives already parsed. A string is still accepted and JSON-parsed to stay
compatible with any legacy Electric row that stored a JSON text column.
*/
export const parseKanbanAssignees = (raw: unknown): string[] => {
	const value = typeof raw === "string" ? safeParse(raw) : raw;

	if (!Array.isArray(value)) {
		return [];
	}

	return value.filter((entry): entry is string => typeof entry === "string");
};

const safeParse = (raw: string): unknown => {
	try {
		return JSON.parse(raw);
	} catch {
		return null;
	}
};
