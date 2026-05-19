//go:build amd64

package embedding

import (
	"fmt"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

const embeddingAVX512MaxULP = 0

func avx512EmbeddingAvailable() bool {
	return cpu.X86.HasAVX512F
}

func randomEmbeddingTable(vocab, hidden int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	table := make([]float32, vocab*hidden)

	for index := range table {
		table[index] = float32((rng.Float64() - 0.5) * 4)
	}

	return table
}

func randomEmbeddingIndices(indexCount, vocab int, seed int64) []int32 {
	rng := rand.New(rand.NewSource(seed))
	indices := make([]int32, indexCount)

	for index := range indices {
		indices[index] = int32(rng.Intn(vocab))
	}

	return indices
}

func TestLookupF32AVX512Parity(t *testing.T) {
	if !avx512EmbeddingAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given Lookup float32 AVX-512", t, func() {
		const vocab = 128
		const indexCount = 7

		for _, hidden := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for hidden=%d", hidden), func() {
				table := randomEmbeddingTable(vocab, hidden, 0xE120+int64(hidden))
				indices := randomEmbeddingIndices(indexCount, vocab, 0xE121+int64(hidden))

				want := make([]float32, indexCount*hidden)
				got := make([]float32, indexCount*hidden)

				runLookupF32Generic(
					unsafe.Pointer(&table[0]),
					unsafe.Pointer(&indices[0]),
					unsafe.Pointer(&want[0]),
					vocab, hidden, indexCount,
				)
				runLookupF32AVX512(
					unsafe.Pointer(&table[0]),
					unsafe.Pointer(&indices[0]),
					unsafe.Pointer(&got[0]),
					vocab, hidden, indexCount,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, embeddingAVX512MaxULP)
			})
		}

		convey.Convey("It should match generic via direct CopyRow asm at parity.Lengths", func() {
			for _, hidden := range parity.Lengths {
				table := randomEmbeddingTable(4, hidden, 0xE122+int64(hidden))
				want := make([]float32, hidden)
				got := make([]float32, hidden)
				copy(want, table[hidden:hidden*2])

				copyRowF32AVX512(&got[0], &table[hidden], hidden)

				parity.AssertFloat32SlicesWithinULP(t, got, want, embeddingAVX512MaxULP)
			}
		})
	})
}

func TestBagF32AVX512Parity(t *testing.T) {
	if !avx512EmbeddingAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given Bag float32 AVX-512", t, func() {
		const vocab = 64
		const bagCount = 3
		const indexCount = 11

		for _, hidden := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match generic for hidden=%d", hidden), func() {
				table := randomEmbeddingTable(vocab, hidden, 0xE130+int64(hidden))
				indices := randomEmbeddingIndices(indexCount, vocab, 0xE131+int64(hidden))
				offsets := []int32{0, 4, 7}

				want := make([]float32, bagCount*hidden)
				got := make([]float32, bagCount*hidden)

				runBagF32Generic(
					unsafe.Pointer(&table[0]),
					unsafe.Pointer(&indices[0]),
					unsafe.Pointer(&offsets[0]),
					unsafe.Pointer(&want[0]),
					vocab, hidden, bagCount, indexCount,
				)
				runBagF32AVX512(
					unsafe.Pointer(&table[0]),
					unsafe.Pointer(&indices[0]),
					unsafe.Pointer(&offsets[0]),
					unsafe.Pointer(&got[0]),
					vocab, hidden, bagCount, indexCount,
				)

				parity.AssertFloat32SlicesWithinULP(t, got, want, embeddingAVX512MaxULP)
			})
		}

		convey.Convey("It should match generic via direct AddRow asm at parity.Lengths", func() {
			for _, hidden := range parity.Lengths {
				table := randomEmbeddingTable(4, hidden, 0xE132+int64(hidden))
				want := make([]float32, hidden)
				got := make([]float32, hidden)

				for dimIndex := range hidden {
					want[dimIndex] = float32(dimIndex) * 0.25
					got[dimIndex] = want[dimIndex]
				}

				for dimIndex := range hidden {
					want[dimIndex] += table[hidden+dimIndex]
				}

				addRowF32AVX512(&got[0], &table[hidden], hidden)

				parity.AssertFloat32SlicesWithinULP(t, got, want, embeddingAVX512MaxULP)
			}
		})
	})
}
