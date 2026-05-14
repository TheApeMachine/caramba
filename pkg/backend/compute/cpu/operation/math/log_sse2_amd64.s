#include "textflag.h"

// logVecSSE2(dst, src []float64) — 2-lane SSE2 implementation.
TEXT ·logVecSSE2(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), AX
	MOVQ src+24(FP), DI
	MOVQ src_len+32(FP), BX

	MOVSD  ·logOne(SB), X10
	SHUFPD $0, X10, X10
	MOVSD  ·logHalf(SB), X11
	SHUFPD $0, X11, X11
	MOVSD  ·logSqrt2(SB), X12
	SHUFPD $0, X12, X12
	MOVSD  ·logLn2(SB), X13
	SHUFPD $0, X13, X13

	CMPQ BX, $2
	JL   done_log_sse2
loop_log_sse2:
	MOVUPD (DI), X0

	// m = (bits & mantmask) | biasOne
	MOVSD  ·logMantMask(SB), X3
	SHUFPD $0, X3, X3
	ANDPD  X0, X3
	MOVSD  ·logBiasOne(SB), X4
	SHUFPD $0, X4, X4
	ORPD   X4, X3                          // X3 = m

	// raw_exp = (bits >> 52) & 0x7FF
	MOVAPD X0, X4
	PSRLQ  $52, X4
	MOVSD  ·logExpMask11(SB), X5
	SHUFPD $0, X5, X5
	PAND   X5, X4
	// raw exponent → double via magic add/sub, then subtract exponent bias.
	MOVSD  ·logMagic52(SB), X5
	SHUFPD $0, X5, X5
	PADDQ  X5, X4
	MOVSD  ·logMagic52D(SB), X5
	SHUFPD $0, X5, X5
	SUBPD  X5, X4
	MOVSD  ·logBias1023D(SB), X5
	SHUFPD $0, X5, X5
	SUBPD  X5, X4                          // X4 = e as double

	// mask = (m > sqrt2)  via  (sqrt2 < m): copy sqrt2 then CMPPD LT
	MOVAPD X12, X5
	CMPPD  X3, X5, $1                      // X5 = (X5 < X3) ? -1 : 0
	// m_corrected = select(mask, m*0.5, m)
	MOVAPD X3, X6
	MULPD  X11, X6                         // m*0.5
	MOVAPD X5, X7
	ANDPD  X6, X7                          // (m*0.5) & mask
	MOVAPD X5, X8
	ANDNPD X3, X8                          // m & ~mask
	ORPD   X8, X7                          // X7 = m_corrected
	MOVAPD X7, X3
	// e_corrected = e + (mask & 1.0)
	ANDPD  X10, X5                         // 1.0 where mask
	ADDPD  X5, X4

	// t = (m - 1) / (m + 1)
	MOVAPD X3, X6
	SUBPD  X10, X6
	MOVAPD X3, X7
	ADDPD  X10, X7
	DIVPD  X7, X6
	MOVAPD X6, X7
	MULPD  X7, X7                          // t^2

	// Horner P(u) = a6 + u*(a5 + u*(... + u*a0))
	MOVSD  ·logA6(SB), X2
	SHUFPD $0, X2, X2
	MULPD  X7, X2
	MOVSD  ·logA5(SB), X8
	SHUFPD $0, X8, X8
	ADDPD  X8, X2

	MULPD  X7, X2
	MOVSD  ·logA4(SB), X8
	SHUFPD $0, X8, X8
	ADDPD  X8, X2

	MULPD  X7, X2
	MOVSD  ·logA3(SB), X8
	SHUFPD $0, X8, X8
	ADDPD  X8, X2

	MULPD  X7, X2
	MOVSD  ·logA2(SB), X8
	SHUFPD $0, X8, X8
	ADDPD  X8, X2

	MULPD  X7, X2
	MOVSD  ·logA1(SB), X8
	SHUFPD $0, X8, X8
	ADDPD  X8, X2

	MULPD  X7, X2
	MOVSD  ·logA0(SB), X8
	SHUFPD $0, X8, X8
	ADDPD  X8, X2                          // X2 = P(t^2)

	MULPD  X6, X2                          // t*P
	ADDPD  X2, X2                          // 2*t*P = log(m)

	MULPD  X13, X4                         // e*ln2
	ADDPD  X4, X2

	MOVUPD X2, (AX)
	ADDQ $16, AX
	ADDQ $16, DI
	SUBQ $2, BX
	CMPQ BX, $2
	JGE  loop_log_sse2

done_log_sse2:
	RET
