package convert

import "errors"

/*
errLenMismatch is returned when dst and src lengths disagree. Wrapped
errors are not part of the public API yet; callers that need
discrimination use errors.Is against this sentinel.
*/
var errLenMismatch = errors.New("convert: destination length does not match source")

/*
ErrLenMismatch is the publicly exported alias of errLenMismatch.
*/
var ErrLenMismatch = errLenMismatch
