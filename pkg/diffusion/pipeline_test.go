package diffusion

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/qpool"
)

func TestPipeline_Generate(test *testing.T) {
	Convey("Given a tiny manifest-backed diffusion pipeline", test, func() {
		previousPublish := qpool.Publish
		qpool.Publish = func(qpool.Event) {}
		defer func() { qpool.Publish = previousPublish }()

		root := test.TempDir()
		outputPath := filepath.Join(root, "out.png")
		tokenizerPath := filepath.Join(root, "tokenizer.json")
		textEncoderManifest := filepath.Join(root, "text-encoder.yml")
		vaeManifest := filepath.Join(root, "vae.yml")
		diffusionManifest := filepath.Join(root, "diffusion.yml")
		textEncoderWeights := filepath.Join(root, "text-encoder.safetensors")
		transformerWeights := filepath.Join(root, "transformer.safetensors")
		vaeWeights := filepath.Join(root, "vae.safetensors")

		So(os.WriteFile(tokenizerPath, tinyTokenizerJSON(), 0o644), ShouldBeNil)
		So(os.WriteFile(textEncoderManifest, []byte(tinyTextEncoderManifest()), 0o644), ShouldBeNil)
		So(os.WriteFile(vaeManifest, []byte(tinyVAEManifest()), 0o644), ShouldBeNil)
		So(os.WriteFile(
			diffusionManifest,
			[]byte(tinyDiffusionManifest(root, textEncoderManifest, vaeManifest, outputPath)),
			0o644,
		), ShouldBeNil)
		So(writeTinySafeTensors(textEncoderWeights, []tinyTensor{{
			name:   "prompt_embeds.weight",
			shape:  []int{1, 7680},
			values: make([]float64, 7680),
		}}), ShouldBeNil)
		So(writeTinySafeTensors(transformerWeights, []tinyTensor{{
			name:   "context_embedder.weight",
			shape:  []int{3, 7680},
			values: make([]float64, 3*7680),
		}}), ShouldBeNil)
		So(writeTinySafeTensors(vaeWeights, nil), ShouldBeNil)

		Convey("It should encode the prompt, denoise, decode through VAE, and write a PNG", func() {
			pipeline, err := NewPipeline(context.Background(), Config{
				Manifest: diffusionManifest,
				Prompt:   "h",
			})
			So(err, ShouldBeNil)
			defer pipeline.Close()

			result, err := pipeline.Generate(context.Background(), "h")
			So(err, ShouldBeNil)
			So(result.Output, ShouldEqual, outputPath)
			So(result.Width, ShouldEqual, 1)
			So(result.Height, ShouldEqual, 1)

			info, err := os.Stat(outputPath)
			So(err, ShouldBeNil)
			So(info.Size() > 0, ShouldBeTrue)
		})
	})
}

func BenchmarkPipeline_Generate(benchmark *testing.B) {
	previousPublish := qpool.Publish
	qpool.Publish = func(qpool.Event) {}
	defer func() { qpool.Publish = previousPublish }()

	root := benchmark.TempDir()
	outputPath := filepath.Join(root, "out.png")
	tokenizerPath := filepath.Join(root, "tokenizer.json")
	textEncoderManifest := filepath.Join(root, "text-encoder.yml")
	vaeManifest := filepath.Join(root, "vae.yml")
	diffusionManifest := filepath.Join(root, "diffusion.yml")
	textEncoderWeights := filepath.Join(root, "text-encoder.safetensors")
	transformerWeights := filepath.Join(root, "transformer.safetensors")
	vaeWeights := filepath.Join(root, "vae.safetensors")

	if err := os.WriteFile(tokenizerPath, tinyTokenizerJSON(), 0o644); err != nil {
		benchmark.Fatal(err)
	}

	if err := os.WriteFile(textEncoderManifest, []byte(tinyTextEncoderManifest()), 0o644); err != nil {
		benchmark.Fatal(err)
	}

	if err := os.WriteFile(vaeManifest, []byte(tinyVAEManifest()), 0o644); err != nil {
		benchmark.Fatal(err)
	}

	if err := os.WriteFile(
		diffusionManifest,
		[]byte(tinyDiffusionManifest(root, textEncoderManifest, vaeManifest, outputPath)),
		0o644,
	); err != nil {
		benchmark.Fatal(err)
	}

	if err := writeTinySafeTensors(textEncoderWeights, []tinyTensor{{
		name:   "prompt_embeds.weight",
		shape:  []int{1, 7680},
		values: make([]float64, 7680),
	}}); err != nil {
		benchmark.Fatal(err)
	}

	if err := writeTinySafeTensors(transformerWeights, []tinyTensor{{
		name:   "context_embedder.weight",
		shape:  []int{3, 7680},
		values: make([]float64, 3*7680),
	}}); err != nil {
		benchmark.Fatal(err)
	}

	if err := writeTinySafeTensors(vaeWeights, nil); err != nil {
		benchmark.Fatal(err)
	}

	pipeline, err := NewPipeline(context.Background(), Config{
		Manifest: diffusionManifest,
		Prompt:   "h",
	})

	if err != nil {
		benchmark.Fatal(err)
	}

	defer pipeline.Close()

	for benchmark.Loop() {
		if _, err := pipeline.Generate(context.Background(), "h"); err != nil {
			benchmark.Fatal(err)
		}
	}
}

type tinyTensor struct {
	name   string
	shape  []int
	values []float64
}

func tinyTokenizerJSON() []byte {
	return []byte(`{
  "version": "1.0",
  "normalizer": null,
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
  "model": {
    "type": "BPE",
    "vocab": {"h": 0},
    "merges": []
  },
  "decoder": {"type": "ByteLevel"},
  "added_tokens": []
}`)
}

func tinyTextEncoderManifest() string {
	return `
system:
  topology:
    inputs: [input_ids]
    nodes:
      - id: prompt_embeds
        op: embedding.token
        in: [input_ids]
        out: [prompt_embeds]
        config: { vocab_size: 1, d_model: 7680 }
`
}

func tinyVAEManifest() string {
	return `
system:
  topology:
    inputs: [latents]
    nodes:
      - id: rgb
        op: shape.reshape
        in: [latents]
        out: [rgb]
        config: { shape: [1, 3, 1, 1] }
`
}

func tinyDiffusionManifest(
	root string,
	textEncoderManifest string,
	vaeManifest string,
	outputPath string,
) string {
	return `
system:
  runtime:
    type: diffusion
    backend: cpu
    model:
      source: ` + root + `
      repo_type: model
    tokenizer:
      source: ` + root + `
      file: tokenizer.json
    text_encoder:
      source: ` + root + `
      manifest: ` + textEncoderManifest + `
      file: text-encoder.safetensors
    transformer:
      source: ` + root + `
      file: transformer.safetensors
    vae:
      source: ` + root + `
      manifest: ` + vaeManifest + `
      file: vae.safetensors
    scheduler:
      type: flow_match_euler_discrete
      num_inference_steps: 1
    generation:
      height: 1
      width: 1
      latent_channels: 3
      latent_downsample: 1
      seed: 1
      pad_token_id: 0
      output: ` + outputPath + `
  topology:
    inputs: [hidden_states, encoder_hidden_states, timestep]
    nodes:
      - id: context_embedder
        op: projection.linear
        in: [encoder_hidden_states]
        out: [context]
        config: { in_features: 7680, out_features: 3 }
      - id: denoise
        op: math.add
        in: [hidden_states, context]
        out: [sample]
`
}

func writeTinySafeTensors(path string, tensors []tinyTensor) error {
	header := make(map[string]any, len(tensors))
	data := make([]byte, 0)

	for _, tensor := range tensors {
		start := len(data)

		for _, value := range tensor.values {
			var encoded [4]byte
			binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(float32(value)))
			data = append(data, encoded[:]...)
		}

		header[tensor.name] = map[string]any{
			"dtype":        "F32",
			"shape":        tensor.shape,
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
