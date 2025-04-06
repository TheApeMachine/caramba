package cmd

import (
	"fmt"
	"runtime"
	"time"

	"github.com/spf13/cobra"
	"github.com/theapemachine/caramba/examples/tui"
	"github.com/theapemachine/caramba/kubrick"
	"github.com/theapemachine/caramba/kubrick/components/spinner"
	"github.com/theapemachine/caramba/kubrick/layouts"
	"github.com/theapemachine/caramba/pkg/errnie"
)

var (
	testCmd = &cobra.Command{
		Use:   "test [type]",
		Short: "Run test scenarios",
		Long:  longTest,
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			switch args[0] {
			case "simd":
				return testSIMD()
			case "tui":
				errnie.SetOutput(false)
				return tui.NewSimple().Run()
			default:
				return fmt.Errorf("unknown test type: %s", args[0])
			}
		},
	}
)

func init() {
	rootCmd.AddCommand(testCmd)
}

var longTest = `
Test the TUI package.

Available test types:
  simd    Test SIMD buffer operations performance
  tui     Test TUI framework with interactive components
`

func testSIMD() error {
	// Enable maximum CPU features
	runtime.GOMAXPROCS(runtime.NumCPU())

	// Create test buffers
	const size = 1024 * 1024 // 1MB of runes
	buf1 := make([]rune, size)
	buf2 := make([]rune, size)
	buf3 := make([]rune, size)

	// Fill buffers with test data
	for i := range buf1 {
		buf1[i] = rune(i % 128)
		buf2[i] = rune(i % 128)
	}

	// Test SIMD buffer comparison
	fmt.Println("Testing buffer comparison...")

	start := time.Now()
	equal := kubrick.CompareBuffers(buf1, buf2)
	simdDuration := time.Since(start)

	// Standard Go comparison
	start = time.Now()
	standardEqual := true
	for i := range buf1 {
		if buf1[i] != buf2[i] {
			standardEqual = false
			break
		}
	}
	standardDuration := time.Since(start)

	fmt.Printf("SIMD comparison: %v in %v\n", equal, simdDuration)
	fmt.Printf("Standard comparison: %v in %v\n", standardEqual, standardDuration)
	fmt.Printf("Speedup: %.2fx\n\n", float64(standardDuration)/float64(simdDuration))

	// Test SIMD buffer clear
	fmt.Println("Testing buffer clear...")

	start = time.Now()
	kubrick.ClearBuffer(buf1, ' ')
	simdDuration = time.Since(start)

	start = time.Now()
	for i := range buf2 {
		buf2[i] = ' '
	}
	standardDuration = time.Since(start)

	fmt.Printf("SIMD clear: %v\n", simdDuration)
	fmt.Printf("Standard clear: %v\n", standardDuration)
	fmt.Printf("Speedup: %.2fx\n\n", float64(standardDuration)/float64(simdDuration))

	// Test SIMD buffer copy
	fmt.Println("Testing buffer copy...")

	start = time.Now()
	kubrick.CopyBuffer(buf3, buf1)
	simdDuration = time.Since(start)

	start = time.Now()
	copy(buf2, buf1)
	standardDuration = time.Since(start)

	fmt.Printf("SIMD copy: %v\n", simdDuration)
	fmt.Printf("Standard copy: %v\n", standardDuration)
	fmt.Printf("Speedup: %.2fx\n", float64(standardDuration)/float64(simdDuration))

	// Test pattern search
	fmt.Println("Testing pattern search...")
	pattern := []rune("Hello, World!")
	// Insert pattern at a random position
	copy(buf1[size/2:], pattern)

	start = time.Now()
	pos := kubrick.FindPattern(buf1, pattern)
	simdDuration = time.Since(start)

	start = time.Now()
	standardPos := -1
	for i := 0; i <= len(buf1)-len(pattern); i++ {
		match := true
		for j := range pattern {
			if buf1[i+j] != pattern[j] {
				match = false
				break
			}
		}
		if match {
			standardPos = i
			break
		}
	}
	standardDuration = time.Since(start)

	fmt.Printf("SIMD pattern search: found at %d in %v\n", pos, simdDuration)
	fmt.Printf("Standard pattern search: found at %d in %v\n", standardPos, standardDuration)
	fmt.Printf("Speedup: %.2fx\n\n", float64(standardDuration)/float64(simdDuration))

	// Test run-length encoding
	fmt.Println("Testing run-length encoding...")
	// Create a buffer with runs of reasonable size
	testSize := size / 10 // Use a smaller test size
	buf1 = make([]rune, testSize)
	for i := range buf1 {
		buf1[i] = rune(i / 100) // Creates runs of 100 identical runes
	}

	start = time.Now()
	runs := kubrick.CountRuns(buf1)
	simdDuration = time.Since(start)

	start = time.Now()
	var standardRuns []int32
	currentRun := int32(1)
	currentValue := buf1[0]
	for i := 1; i < len(buf1); i++ {
		if buf1[i] == currentValue {
			currentRun++
		} else {
			standardRuns = append(standardRuns, currentRun)
			currentRun = 1
			currentValue = buf1[i]
		}
	}
	standardRuns = append(standardRuns, currentRun)
	standardDuration = time.Since(start)

	fmt.Printf("SIMD run-length encoding: %d runs in %v\n", len(runs), simdDuration)
	fmt.Printf("Standard run-length encoding: %d runs in %v\n", len(standardRuns), standardDuration)
	fmt.Printf("Speedup: %.2fx\n\n", float64(standardDuration)/float64(simdDuration))

	// Test run-length expansion
	fmt.Println("Testing run-length expansion...")
	// Use the runs from the previous test
	totalLen := int32(0)
	for _, count := range runs {
		totalLen += count
	}

	// Create properly sized buffers
	expandedBuf := make([]rune, int(totalLen))
	values := make([]rune, len(runs))
	for i := range values {
		values[i] = rune(i)
	}

	start = time.Now()
	expandedLen := kubrick.ExpandRuns(expandedBuf, values, runs)
	simdDuration = time.Since(start)

	start = time.Now()
	pos = 0
	for i := range values {
		for j := int32(0); j < runs[i]; j++ {
			expandedBuf[pos] = values[i]
			pos++
		}
	}
	standardDuration = time.Since(start)

	fmt.Printf("SIMD run-length expansion: %d runes in %v\n", expandedLen, simdDuration)
	fmt.Printf("Standard run-length expansion: %d runes in %v\n", pos, standardDuration)
	fmt.Printf("Speedup: %.2fx\n", float64(standardDuration)/float64(simdDuration))

	// Test buffer differences
	fmt.Println("Testing buffer differences...")

	// Create a buffer with some differences
	copy(buf2, buf1)
	for i := 0; i < 10; i++ {
		pos := (size / 10) * i
		buf2[pos] = 'X'
		buf2[pos+1] = 'Y'
		buf2[pos+2] = 'Z'
	}

	// Test finding first difference
	fmt.Println("\nTesting first difference detection...")

	start = time.Now()
	firstDiff := kubrick.FindFirstDifference(buf1, buf2)
	simdDuration = time.Since(start)

	start = time.Now()
	standardFirstDiff := -1
	for i := range buf1 {
		if buf1[i] != buf2[i] {
			standardFirstDiff = i
			break
		}
	}
	standardDuration = time.Since(start)

	fmt.Printf("SIMD first difference: found at %d in %v\n", firstDiff, simdDuration)
	fmt.Printf("Standard first difference: found at %d in %v\n", standardFirstDiff, standardDuration)
	fmt.Printf("Speedup: %.2fx\n\n", float64(standardDuration)/float64(simdDuration))

	// Test finding all differences
	fmt.Println("Testing full difference detection...")

	start = time.Now()
	diffs := kubrick.FindDifferences(buf1, buf2)
	simdDuration = time.Since(start)

	start = time.Now()
	var standardDiffs []kubrick.DiffResult
	inDiff := false
	diffStart := 0

	for i := range buf1 {
		if buf1[i] != buf2[i] {
			if !inDiff {
				inDiff = true
				diffStart = i
			}
		} else if inDiff {
			standardDiffs = append(standardDiffs, kubrick.DiffResult{
				StartIndex: diffStart,
				Length:     i - diffStart,
				OldRunes:   buf1[diffStart:i],
				NewRunes:   buf2[diffStart:i],
			})
			inDiff = false
		}
	}
	if inDiff {
		standardDiffs = append(standardDiffs, kubrick.DiffResult{
			StartIndex: diffStart,
			Length:     len(buf1) - diffStart,
			OldRunes:   buf1[diffStart:],
			NewRunes:   buf2[diffStart:],
		})
	}
	standardDuration = time.Since(start)

	fmt.Printf("SIMD differences: found %d regions in %v\n", len(diffs), simdDuration)
	fmt.Printf("Standard differences: found %d regions in %v\n", len(standardDiffs), standardDuration)
	fmt.Printf("Speedup: %.2fx\n", float64(standardDuration)/float64(simdDuration))

	return nil
}

func testTUI() {
	// Create a new app with a grid layout containing a spinner
	app := kubrick.NewApp(
		kubrick.WithScreen(
			layouts.NewGridLayout(
				layouts.WithRows(1),
				layouts.WithColumns(1),
				layouts.WithSpacing(1),
				layouts.WithComponents(
					spinner.NewSpinner(
						spinner.WithLabel("Loading system components..."),
					),
				),
			),
		),
	)

	// Simulate some work
	go func() {
		time.Sleep(2 * time.Second)
		app.Write([]byte("Processing..."))

		time.Sleep(2 * time.Second)
		app.Write([]byte("Almost done..."))

		time.Sleep(2 * time.Second)
		app.Close()
	}()

	// Read from app until it's closed
	buf := make([]byte, 1024)
	for {
		_, err := app.Read(buf)
		if err != nil {
			break
		}
	}
}
