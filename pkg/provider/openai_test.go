package provider

import (
	"encoding/json"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/ai"
	"github.com/theapemachine/caramba/pkg/core"
)

// TestNewOpenAIProvider tests the initialization of OpenAIProvider
func TestNewOpenAIProvider(t *testing.T) {
	Convey("Given environment variables for API keys are set", t, func() {
		apiKey := "test-api-key"
		endpoint := "https://api.openai.com/v1"

		Convey("When creating a new OpenAI provider", func() {
			provider := NewOpenAIProvider(apiKey, endpoint)

			Convey("Then it should be properly initialized", func() {
				So(provider, ShouldNotBeNil)
				So(provider.ProviderData, ShouldNotBeNil)
				So(provider.client, ShouldNotBeNil)
				So(provider.enc, ShouldNotBeNil)
				So(provider.dec, ShouldNotBeNil)
			})
		})
	})
}

// TestOpenAIProviderWriteBasics tests basic JSON handling in Write
// This test only verifies that basic byte handling in Write
func TestOpenAIProviderWriteBasics(t *testing.T) {
	Convey("Given an OpenAI provider", t, func() {
		provider := NewOpenAIProvider("test-api-key", "https://api.openai.com/v1")

		Convey("When writing bytes to the provider", func() {
			// Create test data
			testData := &ai.ContextData{
				Messages: []*core.Message{
					core.NewMessage("user", "", "Hello, world!"),
				},
			}

			// Encode test data to bytes
			testBytes, _ := json.Marshal(testData)

			// Create a reader to read the response
			readBuf := make([]byte, 1024)

			// Set up a goroutine to read from the pipe to prevent blocking
			readDone := make(chan bool)
			var readErr error
			var bytesRead int

			go func() {
				// Read from the pipe
				bytesRead, readErr = provider.Read(readBuf)
				readDone <- true
			}()

			// Write to the provider
			n, err := provider.Write(testBytes)

			// Wait for read to complete
			<-readDone

			Convey("Then it should accept the data correctly", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(testBytes))
				// Check read results if available
				if bytesRead > 0 {
					So(readErr, ShouldBeNil)
					So(bytesRead, ShouldBeGreaterThan, 0)
				}
			})
		})
	})
}

// TestOpenAIProviderClose tests the Close method
func TestOpenAIProviderClose(t *testing.T) {
	Convey("Given an OpenAI provider", t, func() {
		provider := NewOpenAIProvider("test-api-key", "https://api.openai.com/v1")

		Convey("When closing the provider", func() {
			err := provider.Close()

			Convey("Then it should close successfully", func() {
				So(err, ShouldBeNil)
				So(provider.ProviderData.Params, ShouldBeNil)
				So(provider.ProviderData.Result, ShouldBeNil)
			})
		})
	})
}

// TestNewOpenAIEmbedder tests the initialization of OpenAIEmbedder
func TestNewOpenAIEmbedder(t *testing.T) {
	Convey("Given API key and endpoint", t, func() {
		apiKey := "test-api-key"
		endpoint := "https://api.openai.com/v1"

		Convey("When creating a new OpenAI embedder", func() {
			embedder := NewOpenAIEmbedder(apiKey, endpoint)

			Convey("Then it should be properly initialized", func() {
				So(embedder, ShouldNotBeNil)
				So(embedder.OpenAIEmbedderData, ShouldNotBeNil)
				So(embedder.apiKey, ShouldEqual, apiKey)
				So(embedder.endpoint, ShouldEqual, endpoint)
				So(embedder.client, ShouldNotBeNil)
				So(embedder.in, ShouldNotBeNil)
				So(embedder.out, ShouldNotBeNil)
				So(embedder.enc, ShouldNotBeNil)
				So(embedder.dec, ShouldNotBeNil)
			})
		})
	})
}

// TestOpenAIEmbedderWriteBasics tests basic JSON handling in Write
// This test only verifies that Write accepts data and returns expected values
// It does not test actual OpenAI API calls
func TestOpenAIEmbedderWriteBasics(t *testing.T) {
	Convey("Given an OpenAI embedder", t, func() {
		embedder := NewOpenAIEmbedder("test-api-key", "https://api.openai.com/v1")

		// Manually set the embedder's client to nil to prevent API calls
		embedder.client = nil

		Convey("When writing valid JSON data", func() {
			msg := core.NewMessage("user", "test", "test content")
			contextData := &ai.ContextData{
				Messages: []*core.Message{msg},
			}
			result := &[]float64{0.1, 0.2, 0.3}
			embedderData := &OpenAIEmbedderData{
				Params: contextData,
				Result: result,
			}

			embedderBytes, _ := json.Marshal(embedderData)
			n, err := embedder.Write(embedderBytes)

			Convey("Then it should accept the data without error", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(embedderBytes))
			})
		})

		Convey("When writing invalid JSON", func() {
			invalidJSON := []byte(`{"broken": "json"`)
			n, err := embedder.Write(invalidJSON)

			Convey("Then it should not fail", func() {
				So(err, ShouldBeNil)
				So(n, ShouldEqual, len(invalidJSON))
			})
		})
	})
}

// TestOpenAIEmbedderClose tests the Close method
func TestOpenAIEmbedderClose(t *testing.T) {
	Convey("Given an OpenAI embedder", t, func() {
		embedder := NewOpenAIEmbedder("test-api-key", "https://api.openai.com/v1")

		Convey("When closing the embedder", func() {
			err := embedder.Close()

			Convey("Then it should close successfully", func() {
				So(err, ShouldBeNil)
				So(embedder.OpenAIEmbedderData.Params, ShouldBeNil)
				So(embedder.OpenAIEmbedderData.Result, ShouldBeNil)
			})
		})
	})
}
