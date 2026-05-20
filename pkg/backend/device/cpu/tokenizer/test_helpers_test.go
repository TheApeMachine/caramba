package tokenizer

import "math/rand"

func randomInt32Vector(count int, seed int64) []int32 {
	randomSource := rand.New(rand.NewSource(seed))
	values := make([]int32, count)

	for index := range values {
		values[index] = int32(randomSource.Intn(1<<20) - (1 << 19))
	}

	return values
}

func assertInt32SliceEqual(
	testingTB interface {
		Helper()
		Fatalf(string, ...any)
	},
	got, want []int32,
) {
	testingTB.Helper()

	if len(got) != len(want) {
		testingTB.Fatalf("length mismatch got=%d want=%d", len(got), len(want))
	}

	for index := range got {
		if got[index] != want[index] {
			testingTB.Fatalf(
				"index %d got=%d want=%d",
				index, got[index], want[index],
			)
		}
	}
}
