package diffusion

/*
Options configures how many denoiser forwards diagnostics run.
*/
type Options struct {
	// IncludeNormTrace runs four additional forwards plus host latent round-trips.
	IncludeNormTrace bool
}

/*
DefaultOptions returns the memory-conscious diagnostic preset (3 denoiser forwards).
*/
func DefaultOptions() Options {
	return Options{
		IncludeNormTrace: false,
	}
}
