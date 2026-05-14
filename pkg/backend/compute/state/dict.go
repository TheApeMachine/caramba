package state

import (
	"fmt"

	"github.com/theapemachine/caramba/pkg/backend/compute/tensor"
)

type StateKeyType uint

type Dict struct {
	tensorBackend tensor.Backend
	err           error
	shape         tensor.Shape
	OpShape       []int
	TargetShape   []int
	Inputs        [][]float64
	Source        string
	File          string
	Cache         string
	Op            string
	At            string
	After         string
	Name          string
	Preset        string
	Pattern       string
	Except        string
	Mode          string
	Targets       []string
	Layer         []float64
	LR            float64
	Beta1         float64
	Beta2         float64
	Eps           float64
	WD            float64 // decoupled weight decay (AdamW when non-zero)
	Momentum      float64
	Alpha         float64
	Rho           float64
	Eta           float64
	LRDecay       float64
	MaxNorm       float64
	Tau           float64
	Theta         float64
	Base          float64
	InitStd       float64
	P             float64
	Nesterov      bool
	Centered      bool
	LineSearch    bool
	Training      bool
	Causal        bool
	Frozen        bool
	Ceil          bool
	CountPad      bool
	HistSize      int
	C1            float64
	Dim           int
	Dim0          int
	Dim1          int
	SplitSize     int
	Window        int
	NumHeads      int
	HeadDim       int
	DModel        int
	VocabSize     int
	InFeatures    int
	OutFeatures   int
	DIn           int
	DQ            int
	DK            int
	DV            int
	InChannels    int
	OutChannels   int
	KernelSize    int
	KernelD       int
	KernelH       int
	KernelW       int
	Stride        int
	StrideD       int
	StrideH       int
	StrideW       int
	Padding       int
	PadD          int
	PadH          int
	PadW          int
	Dilation      int
	DilationD     int
	DilationH     int
	DilationW     int
	Groups        int
	OutPadH       int
	OutPadW       int
	OutH          int
	OutW          int
	Divisor       int
	Reduction     int
	Rank          int
	K             int
	Params        []float64
	Grads         []float64
	Weight        []float64
	Bias          []float64
	Out           []float64
	M, V          []float64
	Buf           []float64
	GradAvg       []float64
	PrevParams    []float64
	PrevGrads     []float64
	SHist         [][]float64
	YHist         [][]float64
	RhoHist       []float64
	Head          int
	Count         int
	Correct       int
	Total         int
	Step          int
	TP            float64
	FP            float64
	FN            float64
	Sum           float64
	X             []float64
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
