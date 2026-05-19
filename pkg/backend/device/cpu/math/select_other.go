//go:build !amd64

package math

var invSqrtDimScaleF32Funcs = []f32InvSqrtDimScaleKernelImpl{
	{InvSqrtDimScaleGeneric, "generic", true},
}

var logSumExpF32Funcs = []f32LogSumExpKernelImpl{
	{LogSumExpGeneric, "generic", true},
}

var outerF32Funcs = []f32OuterKernelImpl{
	{OuterGeneric, "generic", true},
}
