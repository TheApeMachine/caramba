package sampler

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"sync"

	"github.com/theapemachine/caramba/pkg/runtime/program"
)

/*
Categorical is the canonical token sampler runner. It supports
temperature, top-k, top-p (nucleus), and a stop-token list. Each
sampler-id keeps its own derived RNG stream so two samplers in the
same program decode independently from a single program-wide seed.
*/
type Categorical struct {
	mu       sync.Mutex
	baseSeed uint64
	streams  map[string]*rand.ChaCha8
}

/*
New returns a Categorical runner seeded by baseSeed. A baseSeed of
zero is replaced with a fixed value so two unseeded runs are
reproducible.
*/
func New(baseSeed uint64) *Categorical {
	if baseSeed == 0 {
		baseSeed = 0x9e3779b97f4a7c15
	}

	return &Categorical{
		baseSeed: baseSeed,
		streams:  map[string]*rand.ChaCha8{},
	}
}

/*
Next implements op.SamplerRunner.
*/
func (categorical *Categorical) Next(
	execContext context.Context,
	declaration program.SamplerDeclaration,
	logits []float64,
	history []int,
) (int, bool, error) {
	if len(logits) == 0 {
		return 0, false, fmt.Errorf("sampler/categorical: empty logits")
	}

	configuration, err := parseConfig(declaration.Config)

	if err != nil {
		return 0, false, fmt.Errorf("sampler/categorical: %w", err)
	}

	source := categorical.streamFor(declaration.ID)
	stream := rand.New(source)

	token, err := categorical.sample(stream, logits, history, configuration)

	if err != nil {
		return 0, false, err
	}

	stopped := configuration.matchesStop(token)

	if !stopped && configuration.matchesStopSuffix(history, token) {
		stopped = true
	}

	return token, stopped, nil
}

func (categorical *Categorical) sample(
	stream *rand.Rand, logits []float64, history []int, configuration config,
) (int, error) {
	working := append([]float64(nil), logits...)

	applyRepetitionPenalty(working, history, configuration.repetitionPenalty)

	for index, value := range working {
		working[index] = value / configuration.temperature
	}

	if configuration.topK > 0 && configuration.topK < len(working) {
		working = applyTopK(working, configuration.topK)
	}

	probabilities := softmax(working)

	if configuration.topP > 0 && configuration.topP < 1.0 {
		probabilities = applyTopP(probabilities, configuration.topP)
	}

	return drawCategory(stream, probabilities), nil
}

/*
applyRepetitionPenalty shrinks every previously generated token's
logit toward "less likely". Positive logits are divided by the
penalty; negative logits are multiplied. A penalty of 1 (or less)
is a no-op so callers that do not configure repetition can ignore
this entry point.
*/
func applyRepetitionPenalty(logits []float64, history []int, penalty float64) {
	if penalty <= 1 || len(history) == 0 {
		return
	}

	seen := make(map[int]bool, len(history))

	for _, token := range history {
		if token < 0 || token >= len(logits) || seen[token] {
			continue
		}

		seen[token] = true

		if logits[token] < 0 {
			logits[token] *= penalty

			continue
		}

		logits[token] /= penalty
	}
}

func (categorical *Categorical) streamFor(id string) *rand.ChaCha8 {
	categorical.mu.Lock()
	defer categorical.mu.Unlock()

	if stream, ok := categorical.streams[id]; ok {
		return stream
	}

	stream := rand.NewChaCha8(deriveSeed(categorical.baseSeed, id))
	categorical.streams[id] = stream

	return stream
}

func deriveSeed(base uint64, id string) [32]byte {
	mixed := base

	for _, character := range id {
		mixed = mixed*0x100000001b3 ^ uint64(character)
	}

	var seed [32]byte

	for index := 0; index < 4; index++ {
		chunk := mixed ^ (uint64(index+1) * 0x9e3779b97f4a7c15)

		for byteIndex := 0; byteIndex < 8; byteIndex++ {
			seed[index*8+byteIndex] = byte(chunk >> (8 * byteIndex))
		}
	}

	return seed
}

func softmax(values []float64) []float64 {
	maximum := math.Inf(-1)

	for _, value := range values {
		if value > maximum {
			maximum = value
		}
	}

	result := make([]float64, len(values))
	sum := 0.0

	for index, value := range values {
		shifted := value - maximum

		if shifted == math.Inf(-1) {
			result[index] = 0

			continue
		}

		result[index] = math.Exp(shifted)
		sum += result[index]
	}

	if sum == 0 {
		return result
	}

	for index := range result {
		result[index] /= sum
	}

	return result
}

func applyTopK(values []float64, topK int) []float64 {
	indices := make([]int, len(values))

	for index := range indices {
		indices[index] = index
	}

	sort.Slice(indices, func(i, j int) bool { return values[indices[i]] > values[indices[j]] })

	keep := map[int]bool{}

	for index := 0; index < topK; index++ {
		keep[indices[index]] = true
	}

	masked := make([]float64, len(values))

	for index, value := range values {
		if !keep[index] {
			masked[index] = math.Inf(-1)

			continue
		}

		masked[index] = value
	}

	return masked
}

func applyTopP(probabilities []float64, topP float64) []float64 {
	indices := make([]int, len(probabilities))

	for index := range indices {
		indices[index] = index
	}

	sort.Slice(indices, func(i, j int) bool {
		return probabilities[indices[i]] > probabilities[indices[j]]
	})

	cumulative := 0.0
	kept := map[int]bool{}

	for _, index := range indices {
		cumulative += probabilities[index]
		kept[index] = true

		if cumulative >= topP {
			break
		}
	}

	masked := make([]float64, len(probabilities))
	sum := 0.0

	for index, probability := range probabilities {
		if !kept[index] {
			continue
		}

		masked[index] = probability
		sum += probability
	}

	if sum == 0 {
		return probabilities
	}

	for index := range masked {
		masked[index] /= sum
	}

	return masked
}

func drawCategory(stream *rand.Rand, probabilities []float64) int {
	pick := stream.Float64()
	cumulative := 0.0

	for index, probability := range probabilities {
		cumulative += probability

		if pick <= cumulative {
			return index
		}
	}

	return len(probabilities) - 1
}
