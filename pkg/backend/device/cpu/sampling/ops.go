package sampling

import "unsafe"

func GreedySample(logits unsafe.Pointer, vocabSize int) int32 {
	if vocabSize == 0 {
		return 0
	}

	logitView := unsafe.Slice((*float32)(logits), vocabSize)
	return GreedySampleFloat32Native(logitView)
}

func TopKSample(config SamplingConfig, logits unsafe.Pointer, vocabSize int) int32 {
	if vocabSize == 0 {
		return 0
	}

	logitView := unsafe.Slice((*float32)(logits), vocabSize)
	topK := config.TopK

	if topK <= 0 || topK > vocabSize {
		topK = vocabSize
	}

	return TopKSampleFloat32Native(logitView, config.Temperature, topK, config.Seed)
}

func TopPSample(config SamplingConfig, logits unsafe.Pointer, vocabSize int) int32 {
	if vocabSize == 0 {
		return 0
	}

	logitView := unsafe.Slice((*float32)(logits), vocabSize)
	return TopPSampleFloat32Native(logitView, config.Temperature, config.TopP, config.Seed)
}
