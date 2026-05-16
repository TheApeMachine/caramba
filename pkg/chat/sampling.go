package chat

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/theapemachine/caramba/pkg/tokenizer"
)

type generationPolicy struct {
	repetitionPenalty float64
	temperature       float64
	topK              int
	topP              float64
	random            *rand.Rand
	stopTokens        []string
	stopSpecialTokens bool
	stopSequences     [][]int
}

type tokenCandidate struct {
	tokenID int
	logit   float64
	weight  float64
}

func newGenerationPolicy(config ModelConfig) (generationPolicy, error) {
	repetitionPenalty := config.RepetitionPenalty

	if repetitionPenalty == 0 {
		repetitionPenalty = 1.1
	}

	if repetitionPenalty < 1 {
		return generationPolicy{}, fmt.Errorf(
			"chat.model: repetition_penalty must be >= 1",
		)
	}

	if config.Temperature < 0 {
		return generationPolicy{}, fmt.Errorf("chat.model: temperature must be >= 0")
	}

	topP := config.TopP

	if topP == 0 {
		topP = 1
	}

	if topP <= 0 || topP > 1 {
		return generationPolicy{}, fmt.Errorf("chat.model: top_p must be in (0, 1]")
	}

	if config.TopK < 0 {
		return generationPolicy{}, fmt.Errorf("chat.model: top_k must be >= 0")
	}

	return generationPolicy{
		repetitionPenalty: repetitionPenalty,
		temperature:       config.Temperature,
		topK:              config.TopK,
		topP:              topP,
		random:            rand.New(rand.NewSource(config.Seed)), //nolint:gosec
		stopTokens:        append([]string(nil), config.StopTokens...),
		stopSpecialTokens: config.StopSpecialTokens,
	}, nil
}

func (policy *generationPolicy) bindStopSequences(modelTokenizer tokenizer.Tokenizer) error {
	if policy == nil || modelTokenizer == nil {
		return nil
	}

	seen := make(map[string]bool)

	if policy.stopSpecialTokens {
		for _, tokenID := range modelTokenizer.SpecialTokenIDs() {
			policy.addStopSequence([]int{tokenID}, seen)
		}
	}

	for _, stopToken := range policy.stopTokens {
		if stopToken == "" {
			continue
		}

		tokenIDs, err := modelTokenizer.Encode(stopToken)

		if err != nil {
			return fmt.Errorf("chat.model: encode stop token %q: %w", stopToken, err)
		}

		if len(tokenIDs) == 0 {
			return fmt.Errorf("chat.model: stop token %q encoded to no tokens", stopToken)
		}

		policy.addStopSequence(tokenIDs, seen)
	}

	return nil
}

func (policy *generationPolicy) addStopSequence(tokenIDs []int, seen map[string]bool) {
	key := fmt.Sprint(tokenIDs)

	if seen[key] {
		return
	}

	seen[key] = true
	policy.stopSequences = append(policy.stopSequences, append([]int(nil), tokenIDs...))
}

func (policy generationPolicy) stopMatched(tokenIDs []int) bool {
	for _, sequence := range policy.stopSequences {
		if len(sequence) == 0 || len(sequence) > len(tokenIDs) {
			continue
		}

		if tokenSuffixMatches(tokenIDs, sequence) {
			return true
		}
	}

	return false
}

func (policy generationPolicy) stopPending(tokenIDs []int) bool {
	for _, sequence := range policy.stopSequences {
		if len(sequence) <= 1 {
			continue
		}

		prefixLength := commonStopPrefixLength(tokenIDs, sequence)

		if prefixLength > 0 && prefixLength < len(sequence) {
			return true
		}
	}

	return false
}

func commonStopPrefixLength(tokenIDs []int, sequence []int) int {
	limit := min(len(tokenIDs), len(sequence)-1)

	for length := limit; length > 0; length-- {
		if tokenSuffixMatches(tokenIDs, sequence[:length]) {
			return length
		}
	}

	return 0
}

func tokenSuffixMatches(tokenIDs []int, sequence []int) bool {
	offset := len(tokenIDs) - len(sequence)

	for index, tokenID := range sequence {
		if tokenIDs[offset+index] != tokenID {
			return false
		}
	}

	return true
}

func selectLastToken(
	shape []int,
	values []float64,
	tokenIDs []int,
	policy generationPolicy,
) (int, error) {
	if len(shape) == 0 {
		return 0, fmt.Errorf("chat.model: logits shape is required")
	}

	vocabSize := shape[len(shape)-1]

	if vocabSize <= 0 {
		return 0, fmt.Errorf("chat.model: logits vocab dimension must be positive")
	}

	if len(values) < vocabSize {
		return 0, fmt.Errorf(
			"chat.model: logits length %d is smaller than vocab dimension %d",
			len(values), vocabSize,
		)
	}

	logits := values[len(values)-vocabSize:]
	applyRepetitionPenalty(logits, tokenIDs, policy.repetitionPenalty)

	if policy.temperature <= 0 {
		return argmaxToken(logits)
	}

	return sampleToken(logits, policy)
}

func applyRepetitionPenalty(logits []float64, tokenIDs []int, penalty float64) {
	if penalty <= 1 {
		return
	}

	seen := make(map[int]bool, len(tokenIDs))

	for _, tokenID := range tokenIDs {
		if tokenID < 0 || tokenID >= len(logits) || seen[tokenID] {
			continue
		}

		seen[tokenID] = true

		if logits[tokenID] < 0 {
			logits[tokenID] *= penalty
			continue
		}

		logits[tokenID] /= penalty
	}
}

func argmaxToken(logits []float64) (int, error) {
	if len(logits) == 0 {
		return 0, fmt.Errorf("chat.model: logits are empty")
	}

	bestIndex := 0
	bestValue := logits[0]

	for index, value := range logits[1:] {
		if value <= bestValue {
			continue
		}

		bestIndex = index + 1
		bestValue = value
	}

	return bestIndex, nil
}

func sampleToken(logits []float64, policy generationPolicy) (int, error) {
	candidates := rankedCandidates(logits)

	if len(candidates) == 0 {
		return 0, fmt.Errorf("chat.model: logits produced no valid candidates")
	}

	if policy.topK > 0 && policy.topK < len(candidates) {
		candidates = candidates[:policy.topK]
	}

	candidates = weightedCandidates(candidates, policy.temperature)

	if policy.topP < 1 {
		candidates = nucleusCandidates(candidates, policy.topP)
	}

	if len(candidates) == 0 {
		return 0, fmt.Errorf("chat.model: sampling filters removed all candidates")
	}

	return sampleWeighted(candidates, policy.random), nil
}

func rankedCandidates(logits []float64) []tokenCandidate {
	candidates := make([]tokenCandidate, 0, len(logits))

	for tokenID, logit := range logits {
		if math.IsNaN(logit) || math.IsInf(logit, -1) {
			continue
		}

		candidates = append(candidates, tokenCandidate{tokenID: tokenID, logit: logit})
	}

	sort.Slice(candidates, func(leftIndex, rightIndex int) bool {
		left := candidates[leftIndex]
		right := candidates[rightIndex]

		if left.logit == right.logit {
			return left.tokenID < right.tokenID
		}

		return left.logit > right.logit
	})

	return candidates
}

func weightedCandidates(candidates []tokenCandidate, temperature float64) []tokenCandidate {
	if len(candidates) == 0 {
		return candidates
	}

	if math.IsInf(candidates[0].logit, 1) {
		return positiveInfinityCandidates(candidates)
	}

	maxLogit := candidates[0].logit

	for index := range candidates {
		candidates[index].weight = math.Exp((candidates[index].logit - maxLogit) / temperature)
	}

	return candidates
}

func positiveInfinityCandidates(candidates []tokenCandidate) []tokenCandidate {
	out := make([]tokenCandidate, 0)

	for _, candidate := range candidates {
		if !math.IsInf(candidate.logit, 1) {
			break
		}

		candidate.weight = 1
		out = append(out, candidate)
	}

	return out
}

func nucleusCandidates(candidates []tokenCandidate, topP float64) []tokenCandidate {
	total := candidateWeightTotal(candidates)

	if total <= 0 {
		return candidates[:1]
	}

	cumulative := 0.0

	for index, candidate := range candidates {
		cumulative += candidate.weight / total

		if cumulative >= topP {
			return candidates[:index+1]
		}
	}

	return candidates
}

func sampleWeighted(candidates []tokenCandidate, random *rand.Rand) int {
	total := candidateWeightTotal(candidates)

	if total <= 0 || random == nil {
		return candidates[0].tokenID
	}

	threshold := random.Float64() * total
	cumulative := 0.0

	for _, candidate := range candidates {
		cumulative += candidate.weight

		if cumulative >= threshold {
			return candidate.tokenID
		}
	}

	return candidates[len(candidates)-1].tokenID
}

func candidateWeightTotal(candidates []tokenCandidate) float64 {
	total := 0.0

	for _, candidate := range candidates {
		total += candidate.weight
	}

	return total
}
