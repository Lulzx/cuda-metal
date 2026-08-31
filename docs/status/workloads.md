# Downstream workload status

[Status index](../status.md) · [Verification gaps](../known-gaps/verification.md)

Downstream results are pinned, bounded integration gates—not general support for
the parent project.

## Recorded integrations

- **llm.c:** the focused GPT-2 FP32 CUDA test has numerical loss and Apple-GPU
  provenance gates; it is manual/external, not part of the registered CTest set.
- **llama.cpp:** focused SmolLM2 offload sweeps and coherence/provenance checks
  have passed across the recorded NGL range. CuMetal's build disables native
  CUDA FlashAttention where exact coverage is absent.
- **PhysX:** a selected GPU manifest and patched GRB workload have bounded build
  and execution evidence. This is not full PhysX GPU compatibility.
- **HiGHS / cuPDLP-C:** the frozen `afiro` integration exercises cuSPARSE and
  FP64 modes. `wide48` and `ieee64` pass the recorded residual gate; `fast48`
  reaches Optimal but misses its dual-residual threshold.
- **VF64-metal:** the pinned integration script passes `fast48`, `wide48`, and
  `ieee64` on the recorded Apple M4 Pro system. The linked CuMetal support shader
  is checked by exact pin and blob identity.
- **NVIDIA cuda-samples:** all 83 enrolled headless samples pass the current
  manifest. Samples outside that enrollment are unclassified.
- **GROMACS:** the unmodified 2025.4 CUDA path has the recorded deterministic
  CPU-comparison and Apple-GPU provenance gate. A separate same-commit
  comparison against native Metal MR !6137 at
  `c7fc4ef64a23f2fe4795d6342af5bcb769d9ca9a` passes 147 energy comparisons on
  a 96,000-atom water case. On the recorded M4 Pro, rematched three-run warm
  medians are 2.726 ms/step for CuMetal and 2.990 ms/step for native Metal:
  CuMetal is 8.8% lower latency (1.097x throughput) on this bounded case, with
  both restricted to GPU nonbonded/PME and CPU bonded/update. The performance
  TPR uses `nstcalcenergy=5000`; a separate every-step-energy TPR supplies the
  numerical gate. The official GROMACS 96,000-atom water corpus also has a
  matched nonbonded-only comparison against the experimental AdaptiveCpp
  generic/Metal route: three-run warm medians are 5.994 ms/step for CuMetal and
  55.003 ms/step for AdaptiveCpp (9.18x CuMetal throughput), with 147
  deterministic energy comparisons passing at a maximum relative difference
  of 7.99e-06. PME/FFT remained on the CPU for both because GROMACS does not
  currently connect AdaptiveCpp generic/Metal to a GPU FFT library. `ns/day`
  is simulated trajectory throughput and is comparable only for identical TPR
  and task placement. CuMetal currently wins every recorded matched pair, but
  the all-cases goal remains open until native Metal and AdaptiveCpp have been
  run as paired warm series across the enrolled corpus, including a full-GPU
  AdaptiveCpp/Metal FFT route.

Other demonstrations and third-party projects are listed in
[verified results](../verified-results.md). Each claim must retain revision,
command, numerical acceptance, and device provenance where available.
