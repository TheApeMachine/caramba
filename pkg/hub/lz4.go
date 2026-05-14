package hub

import "fmt"

func decodeLZ4Block(src []byte, uncompressedSize int) ([]byte, error) {
	dst := make([]byte, 0, uncompressedSize)
	position := 0

	for position < len(src) {
		token := src[position]
		position++

		literalLength, nextPosition, err := lz4Length(src, position, int(token>>4))

		if err != nil {
			return nil, err
		}

		position = nextPosition

		if position+literalLength > len(src) {
			return nil, fmt.Errorf("lz4 literal length exceeds input")
		}

		dst = append(dst, src[position:position+literalLength]...)
		position += literalLength

		if position == len(src) {
			break
		}

		if position+2 > len(src) {
			return nil, fmt.Errorf("lz4 match offset missing")
		}

		offset := int(src[position]) | int(src[position+1])<<8
		position += 2

		if offset <= 0 || offset > len(dst) {
			return nil, fmt.Errorf("lz4 invalid offset %d", offset)
		}

		matchLength, nextPosition, err := lz4Length(src, position, int(token&0x0f))

		if err != nil {
			return nil, err
		}

		position = nextPosition
		matchLength += 4

		for range matchLength {
			dst = append(dst, dst[len(dst)-offset])
		}
	}

	if len(dst) != uncompressedSize {
		return nil, fmt.Errorf("lz4 size %d != %d", len(dst), uncompressedSize)
	}

	return dst, nil
}

func lz4Length(src []byte, position, value int) (int, int, error) {
	if value != 15 {
		return value, position, nil
	}

	for {
		if position >= len(src) {
			return 0, 0, fmt.Errorf("lz4 length extension exceeds input")
		}

		extension := int(src[position])
		position++
		value += extension

		if extension != 255 {
			return value, position, nil
		}
	}
}

func ungroup4(grouped []byte) []byte {
	length := len(grouped)
	base := length / 4
	remainder := length % 4
	sizes := [4]int{base, base, base, base}

	for index := 0; index < remainder; index++ {
		sizes[index]++
	}

	starts := [4]int{
		0,
		sizes[0],
		sizes[0] + sizes[1],
		sizes[0] + sizes[1] + sizes[2],
	}

	out := make([]byte, length)

	for index := range length {
		group := index % 4
		groupOffset := index / 4
		out[index] = grouped[starts[group]+groupOffset]
	}

	return out
}
