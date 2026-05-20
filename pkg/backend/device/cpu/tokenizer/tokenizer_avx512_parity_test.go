//go:build amd64

package tokenizer

import (
	"fmt"
	"testing"

	"github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/device/cpu/parity"
	"golang.org/x/sys/cpu"
)

func avx512TokenizerAvailable() bool {
	return cpu.X86.HasAVX512F
}

func TestTokenizerPackInt32AVX512Parity(t *testing.T) {
	if !avx512TokenizerAvailable() {
		t.Skip("AVX-512F required")
	}

	convey.Convey("Given TokenizerPackInt32AVX512", t, func() {
		for _, length := range parity.Lengths {
			convey.Convey(fmt.Sprintf("It should match PackInt32Scalar for N=%d", length), func() {
				source := randomInt32Vector(length, 0x2820+int64(length))
				want := make([]int32, length)
				got := make([]int32, length)

				PackInt32Scalar(want, source)
				TokenizerPackInt32AVX512(&got[0], &source[0], length)

				assertInt32SliceEqual(t, got, want)
			})
		}

		convey.Convey("It should match PackInt32Scalar via direct asm at parity.Lengths", func() {
			for _, length := range parity.Lengths {
				source := randomInt32Vector(length, 0x2821+int64(length))
				want := make([]int32, length)
				got := make([]int32, length)

				PackInt32Scalar(want, source)
				TokenizerPackInt32AVX512Asm(&got[0], &source[0], length)

				assertInt32SliceEqual(t, got, want)
			}
		})
	})
}
