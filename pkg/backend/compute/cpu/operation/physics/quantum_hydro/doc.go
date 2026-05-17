/*
Package quantum_hydro provides operations for quantum-hydrodynamic
research workflows (Madelung formulation of the Schrödinger equation,
Bohm quantum potential, diffusion / Fokker–Planck spatial terms, and
related PDE-flavoured physics).

The package currently exposes the discrete Laplacian on uniform 1D / 2D /
3D grids with periodic boundary conditions, implemented across every
backend the platform supports. Thermodynamic experiments (Boltzmann,
Langevin, free energy, contrastive phase) already compose from existing
math primitives via the block templates in pkg/asset/template/block/energy
and are deliberately not duplicated here.

Future additions (each landing as a real implementation on every backend,
no aliasing, no fallbacks):
  - First-derivative stencils (stencil.grad)
  - Divergence stencils (stencil.div)
  - 4th-order Laplacian, anisotropic spacing
  - Dirichlet and Neumann boundary conditions
  - FFT (math.fft) for spectral split-step Schrödinger solvers
*/
package quantum_hydro
