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

Other demonstrations and third-party projects are listed in
[verified results](../verified-results.md). Each claim must retain revision,
command, numerical acceptance, and device provenance where available.
