package devteam

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	openai "github.com/openai/openai-go"
)

const (
	providerChatAttempts = 3
	providerChatTimeout  = 2 * time.Minute
)

func runProviderChat(
	ctx context.Context,
	request func(context.Context) (ChatResponse, error),
) (ChatResponse, error) {
	var lastErr error

	for attempt := 1; attempt <= providerChatAttempts; attempt++ {
		requestCtx, cancel := context.WithTimeout(ctx, providerChatTimeout)
		response, err := request(requestCtx)
		cancel()

		if err == nil {
			return response, nil
		}

		lastErr = err

		if !isTransientProviderError(err) || attempt == providerChatAttempts {
			break
		}

		if err := waitProviderRetry(ctx, attempt); err != nil {
			return ChatResponse{}, errors.Join(lastErr, err)
		}
	}

	return ChatResponse{}, lastErr
}

func waitProviderRetry(ctx context.Context, attempt int) error {
	timer := time.NewTimer(time.Duration(attempt) * 500 * time.Millisecond)
	defer timer.Stop()

	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func isTransientProviderError(err error) bool {
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}

	var openAIError *openai.Error

	if errors.As(err, &openAIError) {
		return transientStatus(openAIError.StatusCode)
	}

	var anthropicError *anthropic.Error

	if errors.As(err, &anthropicError) {
		return transientStatus(anthropicError.StatusCode)
	}

	return false
}

func transientStatus(status int) bool {
	return status == http.StatusRequestTimeout ||
		status == http.StatusConflict ||
		status == http.StatusTooManyRequests ||
		status >= http.StatusInternalServerError
}
