export const parseKanbanAssignees = (raw: string): string[] => {
	try {
		const value: unknown = JSON.parse(raw);

		if (!Array.isArray(value)) {
			return [];
		}

		return value.filter((entry): entry is string => typeof entry === "string");
	} catch {
		return [];
	}
};
