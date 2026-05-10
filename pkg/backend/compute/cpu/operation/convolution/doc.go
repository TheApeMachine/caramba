/*
Package convolution implements Conv1d, Conv2d, Conv3d, and ConvTranspose2d
as operation.Operation implementations for the CPU backend.

The hot inner loop (kernel dot product) is accelerated via AVX2/SSE2 on
amd64 and NEON on arm64.  Weight initialisation uses Kaiming uniform
(He initialisation) with bias initialised uniformly in
[-1/sqrt(fan_in), 1/sqrt(fan_in)].
*/
package convolution
