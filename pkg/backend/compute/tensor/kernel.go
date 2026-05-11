package tensor

/*
Float64ActivationBackend executes activation kernels against resident tensors.
*/
type Float64ActivationBackend interface {
	Backend
	ReLU(x Float64Tensor) (Float64Tensor, error)
	LeakyReLU(x Float64Tensor, alpha float64) (Float64Tensor, error)
	GELU(x Float64Tensor) (Float64Tensor, error)
	Tanh(x Float64Tensor) (Float64Tensor, error)
	Sigmoid(x Float64Tensor) (Float64Tensor, error)
	SwiGLU(x Float64Tensor) (Float64Tensor, error)
}

/*
Float64MathBackend executes math kernels against resident tensors.
*/
type Float64MathBackend interface {
	Backend
	Add(a Float64Tensor, b Float64Tensor) (Float64Tensor, error)
	Mul(a Float64Tensor, b Float64Tensor) (Float64Tensor, error)
	Matmul(lhs Float64Tensor, rhs Float64Tensor) (Float64Tensor, error)
}

/*
Float64FusedBackend executes common fused model kernels against resident tensors.
*/
type Float64FusedBackend interface {
	Backend
	MatmulAdd(a Float64Tensor, b Float64Tensor, bias Float64Tensor) (Float64Tensor, error)
	MatmulAddGELU(a Float64Tensor, b Float64Tensor, bias Float64Tensor) (Float64Tensor, error)
}
