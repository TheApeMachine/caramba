# complete-implementation

You must implement full dtype support, without short-cuts or work-arounds, for the following device backends:

1. CPU
   1. Native Go
   2. SIMD/ASSEMBLY
      1. AVX-512 (You are not required to confirm these work, that will be tested on real amd64 hardware)
      2. AVX2 (You are not required to confirm these work, that will be tested on real amd64 hardware)
      3. SSE2 (You are not required to confirm these work, that will be tested on real amd64 hardware)
      4. NEON (You are required to confirm these work)
2. METAL (You are required to confirm these work)
3. CUDA (You are not required to confirm these work, that will be tested on real amd64 hardware)
4. XLA (You are not required to confirm these work, that will be tested on real amd64 hardware)

You must ensure that all dtype.DType precisions are fully natively supported, and do not under any circumstance try to use a trick like narrow/widen or anything else but full, real, native support.
Always makes sure that you write VECTORIZED, highly optimized code.
When in doubt, read @ARCHITECTURE.md

Make sure that each device backend follows the same structure as the other ones, this is very important.

# RULES

1. You will absolutely not fake implementations
2. You will absolutely not take shortcuts
3. You will absolutely not do anything that is less than optimal implementation
4. You will absolutely not say "I will do only what can be verified on this machine"
5. You will write ALL the optimized code for ALL architectures

Your code will be regularly checked, so any failure to comply will be discovered and you will be made to redo it until it is correct.