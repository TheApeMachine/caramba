package chat

// TEMPORARY DIAGNOSTIC INSTRUMENTATION. Delete this file (and the
// three call sites that reference it in runtime_model_generator.go and
// runtime_hooks.go) once the chat-decode bug is identified. Every print
// is gated by a single package-level bool so toggling it off is a
// one-line change.

import (
	"context"
	"fmt"
	"os"
	"sort"
	"sync/atomic"

	"github.com/theapemachine/caramba/pkg/backend/compute/kv"
	"github.com/theapemachine/caramba/pkg/runtime/op"
	"github.com/theapemachine/caramba/pkg/runtime/program"
)

// chatDebugEnabled gates every diagnostic print in this file. Flip to
// false (or delete the file) to disable.
const chatDebugEnabled = true

var chatDebugStep atomic.Int64

func chatDebugStartup(backend, model string) {
	if !chatDebugEnabled {
		return
	}

	chatDebugStep.Store(0)
	fmt.Fprintf(
		os.Stderr,
		"[chat-debug] startup backend=%q model=%q\n",
		backend, model,
	)
}

func chatDebugPreExecute(positionStart, tokenCount int, cache *kv.Cache) {
	if !chatDebugEnabled {
		return
	}

	step := chatDebugStep.Add(1)
	cacheEpoch := -1

	if cache != nil {
		cacheEpoch = int(cache.Epoch())
	}

	fmt.Fprintf(
		os.Stderr,
		"[chat-debug] step=%d positionStart=%d tokenCount=%d kvCacheEpoch=%d\n",
		step, positionStart, tokenCount, cacheEpoch,
	)
}

// debugSamplerWrapper wraps an op.SamplerRunner so we can log the
// sampled token and the top-3 logit indices+values for each decode.
type debugSamplerWrapper struct {
	inner op.SamplerRunner
}

func newDebugSamplerWrapper(inner op.SamplerRunner) op.SamplerRunner {
	if !chatDebugEnabled {
		return inner
	}

	return &debugSamplerWrapper{inner: inner}
}

func (wrapper *debugSamplerWrapper) Next(
	execContext context.Context,
	samplerDeclaration program.SamplerDeclaration,
	logits []float64,
	history []int,
) (int, bool, error) {
	token, stopped, err := wrapper.inner.Next(execContext, samplerDeclaration, logits, history)

	if err != nil {
		fmt.Fprintf(os.Stderr, "[chat-debug] sample err=%v\n", err)

		return token, stopped, err
	}

	top := topKLogits(logits, 3)
	fmt.Fprintf(
		os.Stderr,
		"[chat-debug] sampled token=%d stopped=%t top3=%v historyLen=%d\n",
		token, stopped, top, len(history),
	)

	return token, stopped, nil
}

type logitEntry struct {
	index int
	value float64
}

func topKLogits(logits []float64, k int) []logitEntry {
	if len(logits) == 0 {
		return nil
	}

	scratch := make([]logitEntry, len(logits))

	for index, value := range logits {
		scratch[index] = logitEntry{index: index, value: value}
	}

	sort.Slice(scratch, func(i, j int) bool {
		return scratch[i].value > scratch[j].value
	})

	if k > len(scratch) {
		k = len(scratch)
	}

	return scratch[:k]
}
