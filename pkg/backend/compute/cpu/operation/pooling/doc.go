/*
Package pooling implements 2-D pooling operations for 4-D tensors [N, C, H, W].

Supported operations:

  - MaxPool2d   — sliding-window maximum
  - AvgPool2d   — sliding-window average
  - AdaptiveAvgPool2d — output-size–driven average pooling
  - AdaptiveMaxPool2d — output-size–driven max pooling

Import as:

	import "github.com/theapemachine/caramba/backend/compute/cpu/operation/pooling"
*/
package pooling
