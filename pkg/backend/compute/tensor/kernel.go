package tensor

/*
Float64ActivationBackend executes activation kernels against resident tensors.
*/
type Float64ActivationBackend interface {
	Backend
	ReLU(Float64Tensor) (Float64Tensor, error)
	LeakyReLU(Float64Tensor, float64) (Float64Tensor, error)
	GELU(Float64Tensor) (Float64Tensor, error)
	Tanh(Float64Tensor) (Float64Tensor, error)
	Sigmoid(Float64Tensor) (Float64Tensor, error)
	SwiGLU(Float64Tensor) (Float64Tensor, error)
}

/*
Float64MathBackend executes math kernels against resident tensors.
*/
type Float64MathBackend interface {
	Backend
	Add(Float64Tensor, Float64Tensor) (Float64Tensor, error)
	Mul(Float64Tensor, Float64Tensor) (Float64Tensor, error)
	Matmul(Float64Tensor, Float64Tensor) (Float64Tensor, error)
}

/*
Float64FusedBackend executes common fused model kernels against resident tensors.
*/
type Float64FusedBackend interface {
	Backend
	MatmulAdd(Float64Tensor, Float64Tensor, Float64Tensor) (Float64Tensor, error)
	MatmulAddGELU(Float64Tensor, Float64Tensor, Float64Tensor) (Float64Tensor, error)
}
