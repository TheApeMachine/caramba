#include "textflag.h"

TEXT ·TestFCMPS(SB), NOSPLIT, $0
	FMOVS $0.0, F14
	FMOVS $-1.0, F0
	FCMPS F0, F14
	BGT pos
	MOVD $1, R0
	RET
pos:
	MOVD $2, R0
	RET
