package tokenizer

import (
	"unicode"
	"unicode/utf8"
)

func splitByteLevel(text string) []string {
	tokens := make([]string, 0)
	index := 0

	for index < len(text) {
		nextIndex, token := nextByteLevelToken(text, index)

		if token != "" {
			tokens = append(tokens, token)
		}

		index = nextIndex
	}

	return tokens
}

func nextByteLevelToken(text string, index int) (int, string) {
	if contraction, ok := nextContraction(text, index); ok {
		return index + len(contraction), contraction
	}

	currentRune, width := utf8.DecodeRuneInString(text[index:])

	if unicode.IsSpace(currentRune) {
		return nextWhitespaceToken(text, index)
	}

	if unicode.IsLetter(currentRune) {
		return consumeClass(text, index+width, index, unicode.IsLetter)
	}

	if unicode.IsNumber(currentRune) {
		return consumeClass(text, index+width, index, unicode.IsNumber)
	}

	return consumePunctuation(text, index+width, index)
}

func nextWhitespaceToken(text string, index int) (int, string) {
	spaceStart := index
	lastSpaceStart := index

	for index < len(text) {
		currentRune, width := utf8.DecodeRuneInString(text[index:])

		if !unicode.IsSpace(currentRune) {
			break
		}

		lastSpaceStart = index
		index += width
	}

	if index >= len(text) {
		return index, text[spaceStart:index]
	}

	if spaceStart < lastSpaceStart {
		return lastSpaceStart, text[spaceStart:lastSpaceStart]
	}

	currentRune, width := utf8.DecodeRuneInString(text[index:])

	if unicode.IsLetter(currentRune) {
		return consumeClass(text, index+width, lastSpaceStart, unicode.IsLetter)
	}

	if unicode.IsNumber(currentRune) {
		return consumeClass(text, index+width, lastSpaceStart, unicode.IsNumber)
	}

	return consumePunctuation(text, index+width, lastSpaceStart)
}

func consumeClass(
	text string,
	index int,
	start int,
	predicate func(rune) bool,
) (int, string) {
	for index < len(text) {
		currentRune, width := utf8.DecodeRuneInString(text[index:])

		if !predicate(currentRune) {
			break
		}

		index += width
	}

	return index, text[start:index]
}

func consumePunctuation(text string, index int, start int) (int, string) {
	for index < len(text) {
		currentRune, width := utf8.DecodeRuneInString(text[index:])

		if unicode.IsSpace(currentRune) ||
			unicode.IsLetter(currentRune) ||
			unicode.IsNumber(currentRune) {
			break
		}

		if _, ok := nextContraction(text, index); ok {
			break
		}

		index += width
	}

	return index, text[start:index]
}

func nextContraction(text string, index int) (string, bool) {
	for _, contraction := range []string{
		"'s",
		"'t",
		"'re",
		"'ve",
		"'m",
		"'ll",
		"'d",
	} {
		if hasPrefixFold(text[index:], contraction) {
			return text[index : index+len(contraction)], true
		}
	}

	return "", false
}

func hasPrefixFold(text string, prefix string) bool {
	if len(text) < len(prefix) {
		return false
	}

	for index := range prefix {
		left := rune(text[index])
		right := rune(prefix[index])

		if unicode.ToLower(left) != unicode.ToLower(right) {
			return false
		}
	}

	return true
}
