package vsa

// l2NormEpsilon guards L2 normalisation against division by zero and near-zero vectors.
// Chosen well above float64 denormal noise; Bundle and applyL2Normalize use the same cutoff.
const l2NormEpsilon = 1e-12
