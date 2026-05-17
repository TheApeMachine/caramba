package state

import (
	"fmt"
	"math"
	"sync"

	"github.com/theapemachine/caramba/pkg/backend/compute/kv"
	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type StateKeyType uint

type Dict struct {
	mu                  sync.Mutex
	tensorBackend       tensor.Backend
	err                 error
	shape               tensor.Shape
	OpShape             []int
	TargetShape         []int
	Inputs              [][]float64
	KVCache             *kv.Cache
	Source              string
	File                string
	Cache               string
	Revision            string
	RepoType            string
	Text                string
	NodeID              string
	Op                  string
	At                  string
	After               string
	Name                string
	Preset              string
	Pattern             string
	Except              string
	Mode                string
	RoPEType            string
	Targets             []string
	Layer               []float64
	LR                  float64
	Beta1               float64
	Beta2               float64
	Eps                 float64
	WD                  float64 // decoupled weight decay (AdamW when non-zero)
	Momentum            float64
	Alpha               float64
	Rho                 float64
	Eta                 float64
	LRDecay             float64
	MaxNorm             float64
	Tau                 float64
	Theta               float64
	Base                float64
	RoPEFactor          float64
	RoPELowFreqFactor   float64
	RoPEHighFreqFactor  float64
	InitStd             float64
	P                   float64
	Nesterov            bool
	Centered            bool
	LineSearch          bool
	Training            bool
	Causal              bool
	Frozen              bool
	Ceil                bool
	CountPad            bool
	Truncate            bool
	SkipSpecialTokens   bool
	HistSize            int
	C1                  float64
	Dim                 int
	Dim0                int
	Dim1                int
	SliceStart          int
	SliceEnd            int
	SplitSize           int
	ScaleH              int
	ScaleW              int
	Window              int
	PositionStart       int
	RoPEOriginalContext int
	NumHeads            int
	NumKVHeads          int
	HeadDim             int
	DModel              int
	VocabSize           int
	InFeatures          int
	OutFeatures         int
	DIn                 int
	DQ                  int
	DK                  int
	DV                  int
	InChannels          int
	OutChannels         int
	KernelSize          int
	KernelD             int
	KernelH             int
	KernelW             int
	Stride              int
	StrideD             int
	StrideH             int
	StrideW             int
	Padding             int
	PadD                int
	PadH                int
	PadW                int
	Dilation            int
	DilationD           int
	DilationH           int
	DilationW           int
	Groups              int
	OutPadH             int
	OutPadW             int
	OutH                int
	OutW                int
	Divisor             int
	Reduction           int
	Rank                int
	K                   int
	Params              []float64
	Grads               []float64
	Weight              []float64
	Bias                []float64
	Out                 []float64
	M, V                []float64
	Buf                 []float64
	GradAvg             []float64
	PrevParams          []float64
	PrevGrads           []float64
	SHist               [][]float64
	YHist               [][]float64
	RhoHist             []float64
	Head                int
	Count               int
	MaxLength           int
	PadTo               int
	PadID               int
	Correct             int
	Total               int
	Step                int
	TP                  float64
	FP                  float64
	FN                  float64
	Sum                 float64
	X                   []float64
}

func (dict *Dict) RecordAccuracy(correct bool) {
	dict.mu.Lock()
	defer dict.mu.Unlock()

	dict.Total++

	if correct {
		dict.Correct++
	}

	dict.EnsureOperationOutLen(1)
	dict.Out[0] = float64(dict.Correct) / float64(dict.Total)
}

func (dict *Dict) RecordPerplexity(loss float64) {
	dict.mu.Lock()
	defer dict.mu.Unlock()

	dict.Total++
	dict.Sum += loss
	dict.EnsureOperationOutLen(1)
	dict.Out[0] = math.Exp(dict.Sum / float64(dict.Total))
}

func (dict *Dict) RecordF1(tp, fp, fn float64) {
	dict.mu.Lock()
	defer dict.mu.Unlock()

	dict.TP += tp
	dict.FP += fp
	dict.FN += fn
	precision := dict.TP / (dict.TP + dict.FP + 1e-9)
	recall := dict.TP / (dict.TP + dict.FN + 1e-9)
	dict.EnsureOperationOutLen(1)
	dict.Out[0] = 2 * precision * recall / (precision + recall + 1e-9)
}

func NewDict(tensorBackends ...tensor.Backend) *Dict {
	tensorBackend := tensor.Backend(tensor.NewHostBackend())

	if len(tensorBackends) > 0 {
		tensorBackend = tensorBackends[0]
	}

	return &Dict{tensorBackend: tensorBackend}
}

func (dict *Dict) WithLR(lr float64) *Dict {
	dict.LR = lr
	return dict
}

func (dict *Dict) WithBeta1(beta1 float64) *Dict {
	dict.Beta1 = beta1
	return dict
}

func (dict *Dict) WithBeta2(beta2 float64) *Dict {
	dict.Beta2 = beta2
	return dict
}

func (dict *Dict) WithEps(eps float64) *Dict {
	dict.Eps = eps
	return dict
}

func (dict *Dict) WithWD(wd float64) *Dict {
	dict.WD = wd
	return dict
}

func (dict *Dict) WithMomentum(momentum float64) *Dict {
	dict.Momentum = momentum
	return dict
}

func (dict *Dict) WithAlpha(alpha float64) *Dict {
	dict.Alpha = alpha
	return dict
}

func (dict *Dict) WithRho(rho float64) *Dict {
	dict.Rho = rho
	return dict
}

func (dict *Dict) WithEta(eta float64) *Dict {
	dict.Eta = eta
	return dict
}

func (dict *Dict) WithLRDecay(lrDecay float64) *Dict {
	dict.LRDecay = lrDecay
	return dict
}

func (dict *Dict) WithMaxNorm(maxNorm float64) *Dict {
	dict.MaxNorm = maxNorm
	return dict
}

func (dict *Dict) WithTau(tau float64) *Dict {
	dict.Tau = tau
	return dict
}

func (dict *Dict) WithNesterov(nesterov bool) *Dict {
	dict.Nesterov = nesterov
	return dict
}

func (dict *Dict) WithCentered(centered bool) *Dict {
	dict.Centered = centered
	return dict
}

func (dict *Dict) WithLineSearch(lineSearch bool) *Dict {
	dict.LineSearch = lineSearch
	return dict
}

func (dict *Dict) WithHistSize(histSize int) *Dict {
	dict.HistSize = histSize
	return dict
}

func (dict *Dict) WithC1(c1 float64) *Dict {
	dict.C1 = c1
	return dict
}

func (dict *Dict) WithParams(params any) *Dict {
	dict.Params, dict.shape, dict.err = dict.float64Values(params)

	return dict
}

func (dict *Dict) WithGrads(grads any) *Dict {
	dict.Grads, _, dict.err = dict.float64Values(grads)

	return dict
}

func (dict *Dict) WithWeight(weight any) *Dict {
	dict.Weight, _, dict.err = dict.float64Values(weight)

	return dict
}

func (dict *Dict) WithBias(bias any) *Dict {
	dict.Bias, _, dict.err = dict.float64Values(bias)

	return dict
}

func (dict *Dict) WithOut(out []float64) *Dict {
	dict.Out = out
	dict.X = out

	return dict
}

func (dict *Dict) WithM(m any) *Dict {
	dict.M, _, dict.err = dict.float64Values(m)

	return dict
}

func (dict *Dict) WithV(v any) *Dict {
	dict.V, _, dict.err = dict.float64Values(v)

	return dict
}

func (dict *Dict) WithStep(step int) *Dict {
	dict.Step = step
	return dict
}

func (dict *Dict) WithX(x []float64) *Dict {
	dict.X = x

	return dict
}

func (dict *Dict) WithShape(shape []int) *Dict {
	dict.OpShape = append(dict.OpShape[:0], shape...)

	return dict
}

func (dict *Dict) WithTargetShape(shape []int) *Dict {
	dict.TargetShape = append(dict.TargetShape[:0], shape...)

	return dict
}

func (dict *Dict) WithInput(input any) *Dict {
	values, shape, err := dict.float64Values(input)

	if err != nil {
		dict.err = err

		return dict
	}

	if len(dict.OpShape) == 0 {
		dict.OpShape = shape.Dims()
	}

	dict.Inputs = append(dict.Inputs[:0], values)

	return dict
}

func (dict *Dict) WithInputs(inputs ...any) *Dict {
	dict.Inputs = dict.Inputs[:0]

	for _, input := range inputs {
		values, shape, err := dict.float64Values(input)

		if err != nil {
			dict.err = err

			return dict
		}

		if len(dict.OpShape) == 0 {
			dict.OpShape = shape.Dims()
		}

		dict.Inputs = append(dict.Inputs, values)
	}

	return dict
}

func (dict *Dict) Err() error {
	return dict.err
}

func (dict *Dict) RequireOperation(name string) error {
	return dict.RequireOperationInputs(name, 1)
}

func (dict *Dict) RequireOperationInputs(name string, count int) error {
	if dict.err != nil {
		return dict.err
	}

	if len(dict.Inputs) < count {
		return fmt.Errorf("%s: at least %d input(s) required", name, count)
	}

	dict.EnsureOperationOut()

	return nil
}

func (dict *Dict) EnsureOperationOut() {
	if len(dict.Inputs) == 0 {
		return
	}

	if len(dict.Out) == len(dict.Inputs[0]) {
		dict.X = dict.Out

		return
	}

	dict.Out = make([]float64, len(dict.Inputs[0]))
	dict.X = dict.Out
}

func (dict *Dict) EnsureOperationOutLen(length int) {
	if len(dict.Out) == length {
		dict.X = dict.Out

		return
	}

	dict.Out = make([]float64, length)
	dict.X = dict.Out
}

func (dict *Dict) OperationShape() []int {
	if len(dict.OpShape) > 0 {
		return append([]int(nil), dict.OpShape...)
	}

	if len(dict.Inputs) == 0 {
		return nil
	}

	return []int{len(dict.Inputs[0])}
}

func (dict *Dict) OperationLastDim() int {
	shape := dict.OperationShape()

	if len(shape) == 0 {
		return 0
	}

	return shape[len(shape)-1]
}

func (dict *Dict) RoPELayout(name string) (
	batch int,
	numHeads int,
	sequenceLength int,
	headDim int,
	err error,
) {
	shape := dict.OperationShape()

	if len(shape) != 4 {
		return 0, 0, 0, 0, fmt.Errorf(
			"%s: expected [batch, num_heads, seq_len, head_dim]; got rank %d",
			name,
			len(shape),
		)
	}

	batch, numHeads, sequenceLength, headDim = shape[0], shape[1], shape[2], shape[3]

	if batch <= 0 || numHeads <= 0 || sequenceLength <= 0 || headDim <= 0 {
		return 0, 0, 0, 0, fmt.Errorf("%s: all shape dimensions must be positive", name)
	}

	if dict.HeadDim != 0 && dict.HeadDim != headDim {
		return 0, 0, 0, 0, fmt.Errorf(
			"%s: head_dim %d does not match shape head dim %d",
			name,
			dict.HeadDim,
			headDim,
		)
	}

	if headDim%2 != 0 {
		return 0, 0, 0, 0, fmt.Errorf("%s: expected even head_dim, got %d", name, headDim)
	}

	return batch, numHeads, sequenceLength, headDim, nil
}

func (dict *Dict) GQALayout(name string) (
	batch int,
	numHeads int,
	numKVHeads int,
	sequenceLength int,
	headDim int,
	err error,
) {
	shape := dict.OperationShape()

	switch len(shape) {
	case 4:
		batch, numHeads, sequenceLength, headDim = shape[0], shape[1], shape[2], shape[3]
		numKVHeads = dict.NumKVHeads

		if numKVHeads <= 0 {
			return 0, 0, 0, 0, 0, fmt.Errorf(
				"%s: num_kv_heads is required for rank 4 GQA input",
				name,
			)
		}
	case 5:
		batch = shape[0]
		numHeads = shape[1]
		numKVHeads = shape[2]
		sequenceLength = shape[3]
		headDim = shape[4]
	default:
		return 0, 0, 0, 0, 0, fmt.Errorf(
			"%s: expected rank 4 or 5, got %d",
			name,
			len(shape),
		)
	}

	if batch <= 0 || numHeads <= 0 || numKVHeads <= 0 || sequenceLength <= 0 || headDim <= 0 {
		return 0, 0, 0, 0, 0, fmt.Errorf("%s: all shape dimensions must be positive", name)
	}

	if dict.NumHeads != 0 && dict.NumHeads != numHeads {
		return 0, 0, 0, 0, 0, fmt.Errorf(
			"%s: num_heads %d does not match shape heads %d",
			name,
			dict.NumHeads,
			numHeads,
		)
	}

	if len(shape) == 5 && dict.NumKVHeads != 0 && dict.NumKVHeads != numKVHeads {
		return 0, 0, 0, 0, 0, fmt.Errorf(
			"%s: num_kv_heads %d does not match layout kv heads %d",
			name,
			dict.NumKVHeads,
			numKVHeads,
		)
	}

	if dict.HeadDim != 0 && dict.HeadDim != headDim {
		return 0, 0, 0, 0, 0, fmt.Errorf(
			"%s: head_dim %d does not match shape head dim %d",
			name,
			dict.HeadDim,
			headDim,
		)
	}

	if numHeads%numKVHeads != 0 {
		return 0, 0, 0, 0, 0, fmt.Errorf("%s: num_heads must be divisible by num_kv_heads", name)
	}

	return batch, numHeads, numKVHeads, sequenceLength, headDim, nil
}

func (dict *Dict) SetOperationOutput(output []float64) *Dict {
	dict.Out = output
	dict.X = output

	return dict
}

func (dict *Dict) RequireReady(name string) error {
	if dict.err != nil {
		return dict.err
	}

	if len(dict.Params) != len(dict.Grads) {
		return fmt.Errorf("%s: params and grads length mismatch", name)
	}

	dict.EnsureOut()

	if len(dict.M) != len(dict.Params) {
		dict.M = make([]float64, len(dict.Params))
	}

	if len(dict.V) != len(dict.Params) {
		dict.V = make([]float64, len(dict.Params))
	}

	return nil
}

func (dict *Dict) EnsureOut() {
	if len(dict.Out) == len(dict.Params) {
		dict.X = dict.Out
		return
	}

	dict.Out = make([]float64, len(dict.Params))
	dict.X = dict.Out
}

func (dict *Dict) EnsureBuf() {
	if len(dict.Buf) == len(dict.Params) {
		return
	}

	dict.Buf = make([]float64, len(dict.Params))
}

func (dict *Dict) EnsureGradAvg() {
	if len(dict.GradAvg) == len(dict.Params) {
		return
	}

	dict.GradAvg = make([]float64, len(dict.Params))
}

func (dict *Dict) EnsureHistory() {
	if dict.HistSize <= 0 {
		dict.HistSize = 10
	}

	if len(dict.SHist) == dict.HistSize && len(dict.YHist) == dict.HistSize &&
		len(dict.RhoHist) == dict.HistSize {
		return
	}

	dict.SHist = make([][]float64, dict.HistSize)
	dict.YHist = make([][]float64, dict.HistSize)
	dict.RhoHist = make([]float64, dict.HistSize)
}

func (dict *Dict) float64Values(value any) ([]float64, tensor.Shape, error) {
	switch typedValue := value.(type) {
	case tensor.Float64Tensor:
		values, err := dict.tensorBackend.DownloadFloat64(typedValue)

		return values, typedValue.Shape(), err
	case []float64:
		shape, err := tensor.NewShape([]int{len(typedValue)})

		return typedValue, shape, err
	default:
		return nil, tensor.Shape{}, fmt.Errorf("state: expected float64 tensor or []float64, got %T", value)
	}
}
