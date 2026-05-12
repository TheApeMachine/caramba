// Bridges PJRT-backed VSA into the package as its own translation unit (cgo
// ignores _*.cpp sources). Keeps static helpers in _vsa_xla.cpp out of
// xla_sources.cpp to avoid name collisions.

#ifdef __XLA_BUILD__
#include "_vsa_xla.cpp"
#endif
