/*
deriveProjectSlug mirrors pkg/backend/api/research_projects.go so the wizard preview
shows the same slug the provision API will persist.
*/
export const deriveProjectSlug = (name: string): string => {
	const normalized = name.trim().toLowerCase();
	let slug = "";
	let lastDash = false;

	for (const character of normalized) {
		const isAlphaNumeric =
			(character >= "a" && character <= "z") ||
			(character >= "0" && character <= "9");

		if (isAlphaNumeric) {
			slug += character;
			lastDash = false;
			continue;
		}

		if (!lastDash && slug.length > 0) {
			slug += "-";
			lastDash = true;
		}
	}

	slug = slug.replace(/^-+|-+$/g, "");

	if (!slug) {
		return "project";
	}

	const maxSlugLength = 64;

	if (slug.length > maxSlugLength) {
		return slug.slice(0, maxSlugLength).replace(/-+$/, "");
	}

	return slug;
};
