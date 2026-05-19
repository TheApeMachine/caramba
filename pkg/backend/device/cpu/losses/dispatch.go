// Package losses implements scalar reduction losses (MSE, MAE, Huber,
// BCE, cross-entropy, KL) for float32, bfloat16, and float16.
//
// MSE and MAE sum kernels follow the pick-at-init model via
// select_{arm64,amd64,386,generic}.go. Float32 accumulation per §5.5
// for mixed-precision registrations.
package losses
