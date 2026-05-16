package chat

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"

	. "github.com/smartystreets/goconvey/convey"
)

func TestNewModelGenerator(test *testing.T) {
	Convey("Given model runtime configuration", test, func() {
		Convey("It should require a positive generation budget", func() {
			manifestPath := filepath.Join(test.TempDir(), "model.yml")
			err := os.WriteFile(manifestPath, []byte(`
inputs:
  - name: input_ids
    type: tensor
system:
  topology:
    nodes:
      - id: gelu
        op: activation.gelu
        in: [input_ids]
        out: [y]
`), 0o644)

			So(err, ShouldBeNil)

			generator, err := NewModelGenerator(
				context.Background(),
				ModelConfig{
					Model:    "openai-community/gpt2",
					Manifest: manifestPath,
				},
			)

			So(generator, ShouldBeNil)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "max_new_tokens")
		})

		Convey("It should require a model source from config or manifest", func() {
			manifestPath := filepath.Join(test.TempDir(), "model.yml")
			err := os.WriteFile(manifestPath, []byte(`
inputs:
  - name: input_ids
    type: tensor
system:
  topology:
    nodes:
      - id: gelu
        op: activation.gelu
        in: [input_ids]
        out: [y]
`), 0o644)

			So(err, ShouldBeNil)

			generator, err := NewModelGenerator(
				context.Background(),
				ModelConfig{
					Manifest:     manifestPath,
					MaxNewTokens: 16,
				},
			)

			So(generator, ShouldBeNil)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "model or tokenizer source is required")
		})

		Convey("It should lower the manifest into compute backend execution", func() {
			root := test.TempDir()
			manifestPath := filepath.Join(root, "model.yml")
			tokenizerPath := filepath.Join(root, "tokenizer.json")
			weightsPath := filepath.Join(root, "model.safetensors")

			err := os.WriteFile(manifestPath, []byte(`
inputs:
  - name: input_ids
    type: tensor
system:
  topology:
    nodes:
      - id: token_embedding
        op: embedding.token
        in: [input_ids]
        out: [hidden]
        config:
          vocab_size: 12
          d_model: 2
`), 0o644)
			So(err, ShouldBeNil)

			err = os.WriteFile(tokenizerPath, testTokenizerJSON(), 0o644)
			So(err, ShouldBeNil)

			err = writeChatSafeTensors(weightsPath, map[string][]float64{
				"token_embedding.weight": {
					0, 1, 2, 3, 4, 5, 6, 7,
					8, 9, 10, 11, 12, 13, 14, 15,
					16, 17, 18, 19, 20, 21, 22, 23,
				},
			}, map[string][]int{
				"token_embedding.weight": {12, 2},
			})
			So(err, ShouldBeNil)

			generator, err := NewModelGenerator(
				context.Background(),
				ModelConfig{
					Model:        root,
					Tokenizer:    root,
					Manifest:     manifestPath,
					MaxNewTokens: 16,
				},
			)
			So(err, ShouldBeNil)
			So(generator.BackendName(), ShouldEqual, string(tensor.Host))

			var response string
			err = generator.Generate(context.Background(), "hello", func(segment string) error {
				response += segment

				return nil
			})

			So(err, ShouldBeNil)
			So(response, ShouldNotBeBlank)
		})

		Convey("It should compile the embedded GPT-2 manifest", func() {
			graph, manifestPath, err := compileModelManifest(DefaultModelManifest)

			So(err, ShouldBeNil)
			So(manifestPath, ShouldEqual, DefaultModelManifest)
			So(graph, ShouldNotBeNil)
			So(graph.Nodes(), ShouldHaveLength, 198)
		})

		Convey("It should compile the embedded Llama manifest shape", func() {
			graph, manifestPath, err := compileModelManifest(
				"model/llm/llama-3-2-1b-instruct.yml",
			)

			So(err, ShouldBeNil)
			So(manifestPath, ShouldEqual, "model/llm/llama-3-2-1b-instruct.yml")
			So(graph, ShouldNotBeNil)
			So(graph.Nodes(), ShouldHaveLength, 291)
		})
	})
}

func TestResolveManifestModelConfig(test *testing.T) {
	Convey("Given manifest-owned model runtime configuration", test, func() {
		Convey("It should read chat settings from the embedded GPT-2 manifest", func() {
			config, err := resolveManifestModelConfig(ModelConfig{
				Manifest: DefaultModelManifest,
			})

			So(err, ShouldBeNil)
			So(config.Runtime, ShouldEqual, "model")
			So(config.Backend, ShouldEqual, "auto")
			So(config.Model, ShouldEqual, "openai-community/gpt2")
			So(config.Tokenizer, ShouldEqual, "openai-community/gpt2")
			So(config.ModelRepoType, ShouldEqual, "model")
			So(config.TokenizerRepoType, ShouldEqual, "model")
			So(config.MaxNewTokens, ShouldEqual, 128)
			So(config.RepetitionPenalty, ShouldEqual, 1.1)
			So(config.Temperature, ShouldEqual, 0.8)
			So(config.TopK, ShouldEqual, 50)
			So(config.TopP, ShouldEqual, 0.95)
			So(config.StopSpecialTokens, ShouldBeTrue)
		})

		Convey("It should keep explicit package config when no runtime block exists", func() {
			manifestPath := filepath.Join(test.TempDir(), "model.yml")
			err := os.WriteFile(manifestPath, []byte(`
inputs:
  - name: input_ids
    type: tensor
system:
  topology:
    nodes:
      - id: gelu
        op: activation.gelu
        in: [input_ids]
        out: [y]
`), 0o644)

			So(err, ShouldBeNil)

			config, err := resolveManifestModelConfig(ModelConfig{
				Model:        "local-model",
				Tokenizer:    "local-tokenizer",
				Backend:      "cpu",
				Manifest:     manifestPath,
				MaxNewTokens: 8,
			})

			So(err, ShouldBeNil)
			So(config.Model, ShouldEqual, "local-model")
			So(config.Tokenizer, ShouldEqual, "local-tokenizer")
			So(config.Backend, ShouldEqual, "cpu")
			So(config.MaxNewTokens, ShouldEqual, 8)
		})
	})
}

func TestSelectLastToken(test *testing.T) {
	Convey("Given last-token logits and a generation policy", test, func() {
		Convey("It should penalize tokens already present in the context", func() {
			tokenID, err := selectLastToken(
				[]int{1, 1, 4},
				[]float64{0, 10, 9, 1},
				[]int{1},
				generationPolicy{repetitionPenalty: 2},
			)

			So(err, ShouldBeNil)
			So(tokenID, ShouldEqual, 2)
		})
	})
}

func TestModelConfig_ComputeBackend(test *testing.T) {
	Convey("Given model backend configuration", test, func() {
		Convey("It should default package callers to CPU", func() {
			backend, err := ModelConfig{}.ComputeBackend()

			So(err, ShouldBeNil)
			So(backend.Location(), ShouldEqual, tensor.Host)
			So(backend.Close(), ShouldBeNil)
		})

		Convey("It should reject unknown backend names", func() {
			backend, err := ModelConfig{Backend: "quantum"}.ComputeBackend()

			So(backend, ShouldBeNil)
			So(err, ShouldNotBeNil)
			So(err.Error(), ShouldContainSubstring, "unsupported backend")
		})
	})
}

func BenchmarkNewModelGenerator(benchmark *testing.B) {
	root := benchmark.TempDir()
	manifestPath := filepath.Join(root, "model.yml")
	tokenizerPath := filepath.Join(root, "tokenizer.json")
	weightsPath := filepath.Join(root, "model.safetensors")

	err := os.WriteFile(manifestPath, []byte(`
inputs:
  - name: input_ids
    type: tensor
system:
  topology:
    nodes:
      - id: token_embedding
        op: embedding.token
        in: [input_ids]
        out: [hidden]
        config:
          vocab_size: 12
          d_model: 2
`), 0o644)

	if err != nil {
		benchmark.Fatal(err)
	}

	err = os.WriteFile(tokenizerPath, testTokenizerJSON(), 0o644)

	if err != nil {
		benchmark.Fatal(err)
	}

	err = writeChatSafeTensors(weightsPath, map[string][]float64{
		"token_embedding.weight": {
			0, 1, 2, 3, 4, 5, 6, 7,
			8, 9, 10, 11, 12, 13, 14, 15,
			16, 17, 18, 19, 20, 21, 22, 23,
		},
	}, map[string][]int{
		"token_embedding.weight": {12, 2},
	})

	if err != nil {
		benchmark.Fatal(err)
	}

	config := ModelConfig{
		Model:        root,
		Tokenizer:    root,
		Manifest:     manifestPath,
		MaxNewTokens: 16,
	}

	for benchmark.Loop() {
		generator, err := NewModelGenerator(context.Background(), config)

		if err != nil {
			benchmark.Fatal(err)
		}

		_ = generator.backend.Close()
	}
}

func BenchmarkModelConfig_ComputeBackend(benchmark *testing.B) {
	config := ModelConfig{Backend: "cpu"}

	for benchmark.Loop() {
		backend, _ := config.ComputeBackend()
		_ = backend.Close()
	}
}

func BenchmarkModelGenerator_BackendName(benchmark *testing.B) {
	backend, err := ModelConfig{Backend: "cpu"}.ComputeBackend()

	if err != nil {
		benchmark.Fatal(err)
	}

	defer func() {
		_ = backend.Close()
	}()

	generator := &ModelGenerator{backend: backend}

	for benchmark.Loop() {
		_ = generator.BackendName()
	}
}

func writeChatSafeTensors(
	path string, tensors map[string][]float64, shapes map[string][]int,
) error {
	header := make(map[string]any, len(tensors))
	data := make([]byte, 0)

	for name, values := range tensors {
		start := len(data)

		for _, value := range values {
			var encoded [4]byte
			binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(float32(value)))
			data = append(data, encoded[:]...)
		}

		header[name] = map[string]any{
			"dtype":        "F32",
			"shape":        shapes[name],
			"data_offsets": []int{start, len(data)},
		}
	}

	headerBytes, err := json.Marshal(header)

	if err != nil {
		return err
	}

	output := make([]byte, 8+len(headerBytes)+len(data))
	binary.LittleEndian.PutUint64(output[:8], uint64(len(headerBytes)))
	copy(output[8:], headerBytes)
	copy(output[8+len(headerBytes):], data)

	return os.WriteFile(path, output, 0o644)
}

func testTokenizerJSON() []byte {
	return []byte(`{
  "version": "1.0",
  "normalizer": null,
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "trim_offsets": true
  },
  "model": {
    "type": "BPE",
    "vocab": {
      "h": 0,
      "e": 1,
      "l": 2,
      "o": 3,
      "he": 4,
      "hel": 5,
      "hell": 6,
      "hello": 7,
      "!": 8,
      "<|endoftext|>": 9,
      "\u00c3": 10,
      "\u00a9": 11
    },
    "merges": [
      "h e",
      "he l",
      "hel l",
      "hell o"
    ]
  },
  "decoder": {
    "type": "ByteLevel"
  },
  "added_tokens": [
    {
      "id": 9,
      "content": "<|endoftext|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
  ]
}`)
}
