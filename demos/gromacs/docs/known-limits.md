# Known limits

[Back to the GROMACS documentation index](../README.md)

- **Only single precision is on the GPU.** Metal has no FP64, so double cuFFT
  entry points retain the CPU implementation. GROMACS's mixed-precision build
  uses the single-precision transforms. Small grids also stay on the CPU when
  dispatch would cost more than the transform; `CUMETAL_FFT_METAL=1` overrides
  this and `CUMETAL_DEBUG_FFT=1` reports the selected path.
- **Single rank, single GPU.** The demo uses `GMX_MPI=OFF`: no domain
  decomposition, halo exchange, or PME/PP split across ranks.
- **Mixed precision only.** GROMACS's double-precision build is untested here.
- **The primary runner covers three systems.** `run.sh` covers villin, rnase,
  and optionally ADH. `bash demos/gromacs/sweep.sh` runs every case in every
  downloaded benchmark archive through the same gate: 82 cases up to 1.07M
  atoms, including reaction-field, virtual-site, CHARMM force-switch,
  pressure-coupled, and pure-water variants. `fetch.sh` downloads archives in
  concurrent byte ranges because the origin throttles each connection.
  Free-energy perturbation remains unexercised.
- **One case is outside its numerical noise floor.** Pure water at 768k atoms
  differs from the CPU build by `2.1e-03` at step 0 against a `1.6e-03` floor.
  The source is nonbonded/PME, not update: moving update to the CPU does not
  change it. STMV shows the same shape in `Coul. recip.` at 2.3 times its floor;
  it is unchanged by either FFT backend but drops 90x with PME on the CPU.
- **Correctness-demo timing is not performance evidence.** The primary gate
  uses Debug-side CuMetal, fftpack for the CPU mesh, and `nstcalcenergy = 1`,
  forcing an energy reduction every step. Use only the warm, Release,
  matched-placement series in [Performance and comparisons](performance.md)
  for throughput claims.

Broader project-wide CUDA gaps, which are not specific to this demo, remain
tracked in [`docs/known-gaps.md`](../../../docs/known-gaps.md).
