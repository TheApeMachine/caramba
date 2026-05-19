package activation

import "unsafe"

func SwiGLUPackedF32Generic(dst, packed *float32, batch, halfCount int) {
	destination := unsafe.Slice(dst, batch*halfCount)
	source := unsafe.Slice(packed, batch*halfCount*2)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		rowBase := batchIndex * halfCount * 2
		dstBase := batchIndex * halfCount
		gate := source[rowBase : rowBase+halfCount]
		up := source[rowBase+halfCount : rowBase+halfCount*2]
		out := destination[dstBase : dstBase+halfCount]
		SwiGLUTensorsF32Generic(&out[0], &gate[0], &up[0], halfCount)
	}
}

func LinGLUPackedF32Generic(dst, packed *float32, batch, halfCount int) {
	destination := unsafe.Slice(dst, batch*halfCount)
	source := unsafe.Slice(packed, batch*halfCount*2)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		rowBase := batchIndex * halfCount * 2
		dstBase := batchIndex * halfCount
		gate := source[rowBase : rowBase+halfCount]
		up := source[rowBase+halfCount : rowBase+halfCount*2]
		out := destination[dstBase : dstBase+halfCount]
		LinGLUTensorsF32Generic(&out[0], &gate[0], &up[0], halfCount)
	}
}

func ReGLUPackedF32Generic(dst, packed *float32, batch, halfCount int) {
	destination := unsafe.Slice(dst, batch*halfCount)
	source := unsafe.Slice(packed, batch*halfCount*2)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		rowBase := batchIndex * halfCount * 2
		dstBase := batchIndex * halfCount
		gate := source[rowBase : rowBase+halfCount]
		up := source[rowBase+halfCount : rowBase+halfCount*2]
		out := destination[dstBase : dstBase+halfCount]
		ReGLUTensorsF32Generic(&out[0], &gate[0], &up[0], halfCount)
	}
}

func GLUPackedF32Generic(dst, packed *float32, batch, halfCount int) {
	destination := unsafe.Slice(dst, batch*halfCount)
	source := unsafe.Slice(packed, batch*halfCount*2)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		rowBase := batchIndex * halfCount * 2
		dstBase := batchIndex * halfCount
		gate := source[rowBase : rowBase+halfCount]
		up := source[rowBase+halfCount : rowBase+halfCount*2]
		out := destination[dstBase : dstBase+halfCount]
		GLUTensorsF32Generic(&out[0], &gate[0], &up[0], halfCount)
	}
}

func SiGLUPackedF32Generic(dst, packed *float32, batch, halfCount int) {
	destination := unsafe.Slice(dst, batch*halfCount)
	source := unsafe.Slice(packed, batch*halfCount*2)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		rowBase := batchIndex * halfCount * 2
		dstBase := batchIndex * halfCount
		gate := source[rowBase : rowBase+halfCount]
		up := source[rowBase+halfCount : rowBase+halfCount*2]
		out := destination[dstBase : dstBase+halfCount]
		SiGLUTensorsF32Generic(&out[0], &gate[0], &up[0], halfCount)
	}
}

func SeGLUPackedF32Generic(dst, packed *float32, batch, halfCount int) {
	destination := unsafe.Slice(dst, batch*halfCount)
	source := unsafe.Slice(packed, batch*halfCount*2)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		rowBase := batchIndex * halfCount * 2
		dstBase := batchIndex * halfCount
		gate := source[rowBase : rowBase+halfCount]
		up := source[rowBase+halfCount : rowBase+halfCount*2]
		out := destination[dstBase : dstBase+halfCount]
		SeGLUTensorsF32Generic(&out[0], &gate[0], &up[0], halfCount)
	}
}

func GeGLUPackedF32Generic(dst, packed *float32, batch, halfCount int) {
	destination := unsafe.Slice(dst, batch*halfCount)
	source := unsafe.Slice(packed, batch*halfCount*2)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		rowBase := batchIndex * halfCount * 2
		dstBase := batchIndex * halfCount
		gate := source[rowBase : rowBase+halfCount]
		up := source[rowBase+halfCount : rowBase+halfCount*2]
		out := destination[dstBase : dstBase+halfCount]
		GeGLUTensorsF32Generic(&out[0], &gate[0], &up[0], halfCount)
	}
}

func GeGLUTanhPackedF32Generic(dst, packed *float32, batch, halfCount int) {
	destination := unsafe.Slice(dst, batch*halfCount)
	source := unsafe.Slice(packed, batch*halfCount*2)

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		rowBase := batchIndex * halfCount * 2
		dstBase := batchIndex * halfCount
		gate := source[rowBase : rowBase+halfCount]
		up := source[rowBase+halfCount : rowBase+halfCount*2]
		out := destination[dstBase : dstBase+halfCount]
		GeGLUTanhTensorsF32Generic(&out[0], &gate[0], &up[0], halfCount)
	}
}
