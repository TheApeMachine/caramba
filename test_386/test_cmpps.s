#include "textflag.h"
TEXT ·Test(SB), NOSPLIT, $0
    CMPPS $6, X3, X2
    RET
