package utils

import (
	"strings"
	"time"

	"github.com/JesusIslam/tldr"
	"github.com/goombaio/namegenerator"
	"github.com/neurosnap/sentences"
	"github.com/theapemachine/caramba/pkg/errnie"
)

func SummarizeText(text string, maxLength int) string {
	// If text is under maxLength characters, use it as-is
	if len(text) <= maxLength {
		return text
	}

	// Use proper sentence tokenization
	tokenizer := sentences.NewSentenceTokenizer(nil)

	// Get all sentences first
	sentences := tokenizer.Tokenize(text)
	totalSentences := len(sentences)

	// Calculate how many sentences to keep based on the character ratio
	proportion := float64(maxLength) / float64(len(text))
	intoSentences := max(int(float64(totalSentences)*proportion), 1)

	// Now get the summary using tldr with the accurate sentence count
	bag := tldr.New()
	result, err := bag.Summarize(text, intoSentences)
	if errnie.New(errnie.WithError(err)) != nil {
		return errnie.New(errnie.WithError(err)).Error()
	}

	// Join the sentences back together with newlines
	return strings.Join(result, "\n")
}

func GenerateName() string {
	seed := time.Now().UTC().UnixNano()
	nameGenerator := namegenerator.NewNameGenerator(seed)

	name := nameGenerator.Generate()

	return name
}
