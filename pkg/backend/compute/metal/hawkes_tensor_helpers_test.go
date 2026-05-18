//go:build darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
	"github.com/theapemachine/caramba/pkg/backend/compute/ir"
	computetensor "github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

const hawkesMultiplier uint64 = 6364136223846793005
const hawkesIncrement uint64 = 1442695040888963407

func hawkesOpsForTest(test testing.TB, tensorBackend *TensorBackend) *MetalHawkes {
	test.Helper()

	hawkesOps, err := tensorBackend.hawkes()
	So(err, ShouldBeNil)

	return hawkesOps
}

func hawkesBenchmarkOps(benchmark *testing.B) (*TensorBackend, *MetalHawkes) {
	benchmark.Helper()

	tensorBackend, err := NewTensorBackend()
	if err != nil {
		benchmark.Skipf("Metal tensor backend unavailable: %v", err)
	}

	hawkesOps, err := tensorBackend.hawkes()
	if err != nil {
		benchmark.Fatal(err)
	}

	return tensorBackend, hawkesOps
}

func hawkesShape(test testing.TB, elementCount int) computetensor.Shape {
	test.Helper()

	shape, err := computetensor.NewShape([]int{elementCount})
	So(err, ShouldBeNil)

	return shape
}

func hawkesBenchmarkShape(benchmark *testing.B, elementCount int) computetensor.Shape {
	benchmark.Helper()

	shape, err := computetensor.NewShape([]int{elementCount})
	if err != nil {
		benchmark.Fatal(err)
	}

	return shape
}

func hawkesTimes(eventCount int) []float64 {
	values := make([]float64, eventCount)

	for index := range values {
		values[index] = float64(float32(float64(index+1) / 10))
	}

	return values
}

func hawkesAlpha(processCount int) []float64 {
	values := make([]float64, processCount)

	for index := range values {
		values[index] = float64(float32(0.2 + float64(index)*0.05))
	}

	return values
}

func hawkesBeta(processCount int) []float64 {
	values := make([]float64, processCount)

	for index := range values {
		values[index] = float64(float32(0.5 + float64(index)*0.1))
	}

	return values
}

func hawkesMu(processCount int) []float64 {
	values := make([]float64, processCount)

	for index := range values {
		values[index] = float64(float32(0.8 + float64(index)*0.1))
	}

	return values
}

func hawkesZeros(elementCount int) []float64 {
	return make([]float64, elementCount)
}

func hawkesIntensities(eventCount int) []float64 {
	values := make([]float64, eventCount)

	for index := range values {
		values[index] = float64(float32(0.75 + float64(index%13)/20))
	}

	return values
}

func referenceHawkesIntensity(
	times []float64,
	alpha []float64,
	beta []float64,
	mu []float64,
	currentTime float64,
) []float64 {
	values := make([]float64, len(alpha))
	current := float32(currentTime)

	for processIndex := range values {
		betaValue := float32(beta[processIndex])
		sum := float32(0)

		for _, eventTime := range times {
			delta := current - float32(eventTime)
			if delta <= 0 {
				break
			}

			sum += float32(math.Exp(float64(-betaValue * delta)))
		}

		values[processIndex] = float64(float32(mu[processIndex]) + float32(alpha[processIndex])*sum)
	}

	return values
}

func referenceHawkesKernelMatrix(times []float64, alpha float64, beta float64) []float64 {
	eventCount := len(times)
	values := make([]float64, eventCount*eventCount)
	alphaValue := float32(alpha)
	betaValue := float32(beta)

	for rowIndex := range eventCount {
		rowTime := float32(times[rowIndex])

		for colIndex := rowIndex + 1; colIndex < eventCount; colIndex++ {
			delta := float32(times[colIndex]) - rowTime
			values[rowIndex*eventCount+colIndex] = float64(
				alphaValue * float32(math.Exp(float64(-betaValue*delta))),
			)
		}
	}

	return values
}

func referenceHawkesLogLikelihood(intensities []float64, integral float64) float64 {
	sum := float32(0)

	for _, intensity := range intensities {
		value := float32(intensity)
		if value <= 0 {
			continue
		}

		sum += float32(math.Log(float64(value)))
	}

	return float64(sum - float32(integral))
}

func referenceHawkesSimulate(
	mu []float64,
	alpha []float64,
	beta []float64,
	tMax float64,
	processCount int,
	maxSteps int,
) []float64 {
	values := make([]float64, processCount*maxSteps)

	for index := range values {
		values[index] = -1
	}

	for processIndex := range processCount {
		referenceHawkesSimulateProcess(values, mu, alpha, beta, float32(tMax), processIndex, maxSteps)
	}

	return values
}

func referenceHawkesSimulateProcess(
	values []float64,
	mu []float64,
	alpha []float64,
	beta []float64,
	tMax float32,
	processIndex int,
	maxSteps int,
) {
	seed := uint64(processIndex+1)*hawkesMultiplier + hawkesIncrement
	eventTime := float32(0)
	count := 0

	for eventTime < tMax && count < maxSteps {
		lambdaStar := referenceHawkesLambda(values, mu, alpha, beta, processIndex, maxSteps, count, eventTime)
		seed = seed*hawkesMultiplier + hawkesIncrement
		uniform := hawkesUniform(seed)
		if uniform < 1e-20 {
			uniform = 1e-20
		}

		eventTime += float32(-math.Log(float64(uniform))) / lambdaStar
		if eventTime >= tMax {
			break
		}

		lambda := referenceHawkesLambda(values, mu, alpha, beta, processIndex, maxSteps, count, eventTime)
		seed = seed*hawkesMultiplier + hawkesIncrement
		if hawkesUniform(seed) <= lambda/lambdaStar {
			values[processIndex*maxSteps+count] = float64(eventTime)
			count++
		}
	}
}

func referenceHawkesLambda(
	values []float64,
	mu []float64,
	alpha []float64,
	beta []float64,
	processIndex int,
	maxSteps int,
	count int,
	eventTime float32,
) float32 {
	lambda := float32(mu[processIndex])

	for eventIndex := range count {
		previous := float32(values[processIndex*maxSteps+eventIndex])
		delta := eventTime - previous
		lambda += float32(alpha[processIndex]) *
			float32(math.Exp(float64(-float32(beta[processIndex])*delta)))
	}

	return lambda
}

func hawkesUniform(seed uint64) float32 {
	return float32((seed>>11)&0xFFFFFF) * (1.0 / 16777216.0)
}

func hawkesGraph(test testing.TB) (
	*ir.Graph,
	[]*ir.Node,
	int64,
	map[string][]float64,
) {
	test.Helper()

	processCount, eventCount, maxSteps := 2, 7, 7
	times := hawkesTimes(eventCount)
	alpha := hawkesAlpha(processCount)
	simAlpha := hawkesZeros(processCount)
	beta := hawkesBeta(processCount)
	mu := hawkesMu(processCount)
	currentTime := []float64{1.2}
	intensities := hawkesIntensities(eventCount)
	integral := []float64{1.25}
	tMax := []float64{2}
	inputs := hawkesGraphInputs(test, times, alpha, simAlpha, beta, mu, currentTime, intensities, integral, tMax)

	intensityNode := hawkesNode(
		"hawkes_intensity",
		"hawkes.intensity",
		hawkesShape(test, processCount),
		inputs[0],
		inputs[1],
		inputs[3],
		inputs[4],
		inputs[5],
	)
	kernelNode := hawkesNode("hawkes_kernel", "hawkes.kernel_matrix", hawkesShape(test, eventCount*eventCount), inputs[0], inputs[1], inputs[3])
	logNode := hawkesNode("hawkes_log", "hawkes.log_likelihood", hawkesShape(test, 1), inputs[0], inputs[6], inputs[7])
	simNode := hawkesNode("hawkes_sim", "hawkes.simulate", hawkesShape(test, processCount*maxSteps), inputs[4], inputs[2], inputs[3], inputs[8])
	graph := ir.NewGraph()

	for _, node := range append(inputs, intensityNode, kernelNode, logNode, simNode) {
		graph.AddNode(node)
	}

	expected := map[string][]float64{
		"hawkes_intensity": referenceHawkesIntensity(times, alpha, beta, mu, currentTime[0]),
		"hawkes_kernel":    referenceHawkesKernelMatrix(times, alpha[0], beta[0]),
		"hawkes_log":       []float64{referenceHawkesLogLikelihood(intensities, integral[0])},
		"hawkes_sim":       referenceHawkesSimulate(mu, simAlpha, beta, tMax[0], processCount, maxSteps),
	}
	expectedBytes := int64(
		(len(times) + len(alpha) + len(simAlpha) + len(beta) + len(mu) +
			len(currentTime) + len(intensities) + len(integral) + len(tMax)) * 4,
	)

	return graph, []*ir.Node{intensityNode, kernelNode, logNode, simNode}, expectedBytes, expected
}

func hawkesGraphInputs(test testing.TB, values ...[]float64) []*ir.Node {
	test.Helper()

	nodes := make([]*ir.Node, len(values))

	for index, value := range values {
		shape := hawkesShape(test, len(value))
		node := ir.NewNode(hawkesInputName(index), ir.OpInput, shape)
		node.SetMetadata("values", value)
		nodes[index] = node
	}

	return nodes
}

func hawkesInputName(index int) string {
	names := []string{"times", "alpha", "sim_alpha", "beta", "mu", "time", "intensities", "integral", "tmax"}

	return names[index]
}

func hawkesNode(
	name string,
	operation ir.OpType,
	shape computetensor.Shape,
	inputs ...*ir.Node,
) *ir.Node {
	node := ir.NewNode(name, operation, shape)

	for _, input := range inputs {
		node.AddInput(input)
	}

	return node
}

func assertHawkesGraphOutputs(
	results map[string]computetensor.Tensor,
	expected map[string][]float64,
) {
	for name, output := range results {
		SoMsg(name+" location", output.Location(), ShouldEqual, computetensor.Metal)
		defer func(value computetensor.Tensor) {
			So(value.Close(), ShouldBeNil)
		}(output)

		values, err := tensorFloat64Values(output)
		SoMsg(
			fmt.Sprintf("%s clone shape_len=%d", name, output.Shape().Len()),
			err,
			ShouldBeNil,
		)
		assertMetalMaxDiff(values, expected[name], hawkesGraphTolerance(name))
	}
}

func hawkesGraphTolerance(name string) float64 {
	if name == "hawkes_log" {
		return 1e-4
	}

	if name == "hawkes_sim" {
		return 2e-5
	}

	return 1e-5
}
