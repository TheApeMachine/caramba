package activation

var (
	expF16LUT, expBF16LUT                 [65536]uint16
	logF16LUT, logBF16LUT                 [65536]uint16
	log1pF16LUT, log1pBF16LUT             [65536]uint16
	expm1F16LUT, expm1BF16LUT             [65536]uint16
	sigmoidF16LUT, sigmoidBF16LUT         [65536]uint16
	logSigmoidF16LUT, logSigmoidBF16LUT   [65536]uint16
	tanhF16LUT, tanhBF16LUT               [65536]uint16
	siluF16LUT, siluBF16LUT               [65536]uint16
	geluTanhF16LUT, geluTanhBF16LUT       [65536]uint16
	geluF16LUT, geluBF16LUT               [65536]uint16
	reluF16LUT, reluBF16LUT               [65536]uint16
	leakyReluF16LUT, leakyReluBF16LUT     [65536]uint16
	eluF16LUT, eluBF16LUT                 [65536]uint16
	celuF16LUT, celuBF16LUT               [65536]uint16
	seluF16LUT, seluBF16LUT               [65536]uint16
	softplusF16LUT, softplusBF16LUT       [65536]uint16
	mishF16LUT, mishBF16LUT               [65536]uint16
	softsignF16LUT, softsignBF16LUT       [65536]uint16
	hardSigmoidF16LUT, hardSigmoidBF16LUT [65536]uint16
	hardSwishF16LUT, hardSwishBF16LUT     [65536]uint16
	hardTanhF16LUT, hardTanhBF16LUT       [65536]uint16
	hardGeluF16LUT, hardGeluBF16LUT       [65536]uint16
	quickGeluF16LUT, quickGeluBF16LUT     [65536]uint16
	tanhShrinkF16LUT, tanhShrinkBF16LUT   [65536]uint16
)
