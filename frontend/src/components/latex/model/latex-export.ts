import type {
	HeadingLevel,
	PaperBlock,
	PaperMetadata,
} from "#/components/latex/model/types";

export function escapeLatexText(text: string): string {
	return text
		.replaceAll("\\", "\\textbackslash{}")
		.replaceAll("{", "\\{")
		.replaceAll("}", "\\}")
		.replaceAll("$", "\\$")
		.replaceAll("&", "\\&")
		.replaceAll("%", "\\%")
		.replaceAll("#", "\\#")
		.replaceAll("_", "\\_")
		.replaceAll("^", "\\textasciicircum{}")
		.replaceAll("~", "\\textasciitilde{}");
}

type InlineToken =
	| { type: "text"; text: string }
	| { type: "bold"; text: string }
	| { type: "italic"; text: string }
	| { type: "code"; text: string }
	| { type: "math"; text: string };

const inlinePatterns: { type: InlineToken["type"]; pattern: RegExp }[] = [
	{ type: "bold", pattern: /^\*\*([^*\n]+)\*\*/ },
	{ type: "code", pattern: /^`([^`\n]+)`/ },
	{ type: "math", pattern: /^\$([^$\n]+)\$/ },
	{ type: "italic", pattern: /^\*([^*\n]+)\*/ },
];

function tokenizeInline(line: string): InlineToken[] {
	const tokens: InlineToken[] = [];
	let cursor = 0;

	while (cursor < line.length) {
		const rest = line.slice(cursor);
		let matched = false;

		for (const { type, pattern } of inlinePatterns) {
			const match = rest.match(pattern);

			if (!match) {
				continue;
			}

			tokens.push({ type, text: match[1] } as InlineToken);
			cursor += match[0].length;
			matched = true;
			break;
		}

		if (matched) {
			continue;
		}

		const last = tokens[tokens.length - 1];

		if (last && last.type === "text") {
			last.text += line[cursor];
		} else {
			tokens.push({ type: "text", text: line[cursor] });
		}

		cursor += 1;
	}

	return tokens;
}

function tokenToLatex(token: InlineToken): string {
	if (token.type === "text") {
		return escapeLatexText(token.text);
	}

	if (token.type === "bold") {
		return `\\textbf{${escapeLatexText(token.text)}}`;
	}

	if (token.type === "italic") {
		return `\\emph{${escapeLatexText(token.text)}}`;
	}

	if (token.type === "code") {
		return `\\texttt{${escapeLatexText(token.text)}}`;
	}

	return `$${token.text}$`;
}

function renderInline(line: string): string {
	return tokenizeInline(line).map(tokenToLatex).join("");
}

function headingCommand(level: HeadingLevel): string {
	if (level === 1) {
		return "section";
	}

	if (level === 2) {
		return "subsection";
	}

	return "subsubsection";
}

function listItems(text: string): string[] {
	return text
		.split("\n")
		.map((line) => line.trim())
		.filter(Boolean);
}

function blockToLatex(block: PaperBlock): string {
	if (block.type === "heading") {
		const trimmed = block.text.trim();
		const escaped = escapeLatexText(trimmed);

		if (block.presentation === "references") {
			return `\\section*{${escaped || "References"}}`;
		}

		if (block.presentation === "acknowledgments") {
			return `\\section*{${escaped || "Acknowledgments"}}`;
		}

		if (!escaped) {
			return `% empty ${headingCommand(block.level)}`;
		}

		return `\\${headingCommand(block.level)}{${escaped}}`;
	}

	if (block.type === "equation") {
		return block.display ? `\\[\n${block.latex}\n\\]` : `$${block.latex}$`;
	}

	if (block.type === "list") {
		const items = listItems(block.text);

		if (items.length === 0) {
			return "";
		}

		const env = block.ordered ? "enumerate" : "itemize";
		const body = items
			.map((item) => `  \\item ${renderInline(item)}`)
			.join("\n");

		return `\\begin{${env}}\n${body}\n\\end{${env}}`;
	}

	const lines = block.text
		.split("\n")
		.map((line) => renderInline(line.trim()))
		.filter(Boolean);

	if (lines.length === 0) {
		return "";
	}

	return `${lines.join("\\\\\n")}\n\n`;
}

function stripAbstractSection(blocks: PaperBlock[]): {
	bodyBlocks: PaperBlock[];
	literalAbstract: string | null;
} {
	const abstractIndex = blocks.findIndex(
		(b) => b.type === "heading" && b.presentation === "abstract",
	);

	if (abstractIndex === -1) {
		return { bodyBlocks: blocks, literalAbstract: null };
	}

	const following = blocks.slice(abstractIndex + 1);
	const nextHeadingAt = following.findIndex((b) => b.type === "heading");
	const abstractTail =
		nextHeadingAt === -1 ? following : following.slice(0, nextHeadingAt);

	const literalAbstract = abstractTail
		.map((b) => blockToLatex(b))
		.filter(Boolean)
		.join("\n\n");

	const skipCount = 1 + abstractTail.length;

	const bodyBlocks = [
		...blocks.slice(0, abstractIndex),
		...blocks.slice(abstractIndex + skipCount),
	];

	return { bodyBlocks, literalAbstract };
}

export function exportPaperToLatex(
	meta: PaperMetadata,
	blocks: PaperBlock[],
): string {
	const title = escapeLatexText(meta.title.trim());
	const authorsRaw = meta.authors.trim();
	const authors = escapeLatexText(authorsRaw.replaceAll("\n", ", "));
	const metaAbstract = escapeLatexText(meta.abstract.trim());
	const keywords = escapeLatexText(meta.keywords.trim());

	const { bodyBlocks, literalAbstract } = stripAbstractSection(blocks);
	const body = bodyBlocks.map(blockToLatex).filter(Boolean).join("\n");

	const lines: string[] = [
		"% Generated by Caramba — safe to compile on the server",
		"\\documentclass{article}",
		"\\usepackage[utf8]{inputenc}",
		"\\begin{document}",
		"",
	];

	if (title) {
		lines.push(`\\title{${title}}`, "");
	}

	if (authors) {
		lines.push(`\\author{${authors}}`, "");
	}

	if (title || authors) {
		lines.push("\\maketitle", "");
	}

	const hasStructuredAbstract = literalAbstract !== null;
	const abstractBody = hasStructuredAbstract ? literalAbstract : metaAbstract;

	if (hasStructuredAbstract || metaAbstract) {
		lines.push("\\begin{abstract}", abstractBody, "\\end{abstract}", "");
	}

	if (keywords) {
		lines.push(`\\textbf{Keywords:} ${keywords}`, "");
	}

	lines.push(body, "\\end{document}", "");

	return lines.join("\n");
}
