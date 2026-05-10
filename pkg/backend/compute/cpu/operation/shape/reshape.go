package shape

// Reshape returns a flat copy of data[0] with no structural change.
// The caller uses TargetShape to interpret the output.
//
// Forward(shape, data[0]) -> copy of data[0]
// shape is the input shape (metadata only); the output has the same number
// of elements laid out identically — only the logical shape changes.
type Reshape struct {
	TargetShape []int
}

// NewReshape creates a Reshape operation whose output is interpreted as
// targetShape.
func NewReshape(targetShape []int) *Reshape {
	return &Reshape{TargetShape: targetShape}
}

// Forward returns a copy of data[0].  shape is accepted for interface
// compatibility but is not used.
func (r *Reshape) Forward(shape []int, data ...[]float64) []float64 {
	src := data[0]
	dst := make([]float64, len(src))
	copy(dst, src)
	return dst
}
