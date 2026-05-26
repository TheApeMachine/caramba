/*
deriveSlug turns a display name into a URL-safe slug. Mirrors the Go
implementation in pkg/research/projects.go so client-derived slugs match
what the server would compute. The server retains authority over
collision resolution (it appends -2, -3, … as needed).
*/
const MAX_SLUG_LENGTH = 64;

export const deriveSlug = (input: string): string => {
	const normalized = input.normalize("NFKD").toLowerCase().trim();

	let lastDash = false;
	let builder = "";

	for (const character of normalized) {
		const code = character.codePointAt(0) ?? 0;
		const isLetter =
			(code >= 0x61 && code <= 0x7a) || /\p{L}/u.test(character);
		const isDigit = (code >= 0x30 && code <= 0x39) || /\p{N}/u.test(character);

		if (isLetter || isDigit) {
			builder += character;
			lastDash = false;
			continue;
		}

		if (!lastDash && builder.length > 0) {
			builder += "-";
			lastDash = true;
		}
	}

	let slug = builder.replace(/^-+|-+$/g, "");

	if (slug === "") {
		return "project";
	}

	if (slug.length > MAX_SLUG_LENGTH) {
		slug = slug.slice(0, MAX_SLUG_LENGTH).replace(/-+$/g, "");
	}

	return slug;
};
