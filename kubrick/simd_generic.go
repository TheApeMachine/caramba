//go:build (!amd64 && !arm64) || nosimd
// +build !amd64,!arm64 nosimd

package kubrick

// CompareBuffers compares two rune buffers.
// Returns true if the buffers are identical.
func CompareBuffers(a, b []rune) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ClearBuffer sets all runes in the buffer to the given value.
func ClearBuffer(buf []rune, value rune) {
	for i := range buf {
		buf[i] = value
	}
}

// CopyBuffer copies src to dst.
func CopyBuffer(dst, src []rune) {
	copy(dst, src)
}

// FindPattern searches for a pattern in the buffer.
// Returns the index of the first occurrence or -1 if not found.
func FindPattern(buf []rune, pattern []rune) int {
	if len(pattern) > len(buf) {
		return -1
	}
	for i := 0; i <= len(buf)-len(pattern); i++ {
		match := true
		for j := range pattern {
			if buf[i+j] != pattern[j] {
				match = false
				break
			}
		}
		if match {
			return i
		}
	}
	return -1
}

// CountRuns counts consecutive identical runes.
// Returns a slice of run lengths.
func CountRuns(buf []rune) []int32 {
	if len(buf) == 0 {
		return nil
	}
	var runs []int32
	currentRun := int32(1)
	currentValue := buf[0]

	for i := 1; i < len(buf); i++ {
		if buf[i] == currentValue {
			currentRun++
		} else {
			runs = append(runs, currentRun)
			currentRun = 1
			currentValue = buf[i]
		}
	}
	runs = append(runs, currentRun)
	return runs
}

// ExpandRuns expands run-length encoded data.
// Takes pairs of (rune, count) and expands them into a buffer.
func ExpandRuns(dst []rune, values []rune, counts []int32) int {
	pos := 0
	for i := range values {
		for j := int32(0); j < counts[i] && pos < len(dst); j++ {
			dst[pos] = values[i]
			pos++
		}
	}
	return pos
}

// DiffResult represents a difference between two buffers
type DiffResult struct {
	StartIndex int    // Start index of the difference
	Length     int    // Length of the different region
	OldRunes   []rune // Original runes in the region
	NewRunes   []rune // New runes in the region
}

// FindDifferences finds all differences between two buffers.
// Returns a slice of DiffResult containing the differences.
func FindDifferences(old, new []rune) []DiffResult {
	var diffs []DiffResult
	inDiff := false
	diffStart := 0

	minLen := min(len(old), len(new))

	for i := range minLen {
		if old[i] != new[i] {
			if !inDiff {
				inDiff = true
				diffStart = i
			}
		} else if inDiff {
			diffs = append(diffs, DiffResult{
				StartIndex: diffStart,
				Length:     i - diffStart,
				OldRunes:   old[diffStart:i],
				NewRunes:   new[diffStart:i],
			})
			inDiff = false
		}
	}

	if inDiff {
		diffs = append(diffs, DiffResult{
			StartIndex: diffStart,
			Length:     minLen - diffStart,
			OldRunes:   old[diffStart:minLen],
			NewRunes:   new[diffStart:minLen],
		})
	}

	// Handle case where buffers have different lengths
	if len(old) != len(new) {
		diffs = append(diffs, DiffResult{
			StartIndex: minLen,
			Length:     abs(len(old) - len(new)),
			OldRunes:   old[minLen:],
			NewRunes:   new[minLen:],
		})
	}

	return diffs
}

// FindFirstDifference finds the first difference between two buffers.
// Returns the index of the first difference, or -1 if buffers are identical.
func FindFirstDifference(old, new []rune) int {
	minLen := min(len(old), len(new))

	for i := range minLen {
		if old[i] != new[i] {
			return i
		}
	}

	if len(old) != len(new) {
		return minLen
	}

	return -1
}

// abs returns the absolute value of x
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
