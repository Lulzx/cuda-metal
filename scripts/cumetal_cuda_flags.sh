#!/usr/bin/env bash
# Shared CUDA host-compile flags for CuMetal's clang+fatbin shim toolchain.
# Source from build scripts:  source "$(dirname "$0")/cumetal_cuda_flags.sh"

# Resolve GPU arch (override with CUMETAL_CUDA_ARCH / CUMETAL_LLMC_CUDA_ARCH).
cumetal_cuda_arch() {
    if [[ -n "${CUMETAL_CUDA_ARCH:-}" ]]; then
        echo "${CUMETAL_CUDA_ARCH}"
    elif [[ -n "${CUMETAL_LLMC_CUDA_ARCH:-}" ]]; then
        echo "${CUMETAL_LLMC_CUDA_ARCH}"
    else
        echo "sm_80"
    fi
}

# Extra -Xclang target-feature flags so Homebrew LLVM (PTX 4.2 default) can target
# newer SM versions. Without these, --cuda-gpu-arch=sm_80 fails at compile time.
cumetal_cuda_ptx_feature_flags() {
    local arch="$1"
    case "${arch}" in
        sm_90*|sm_89|sm_86|sm_80)
            echo -Xclang -target-feature -Xclang +ptx70
            ;;
        sm_78|sm_75)
            echo -Xclang -target-feature -Xclang +ptx63
            ;;
        sm_72|sm_70)
            echo -Xclang -target-feature -Xclang +ptx60
            ;;
        sm_61)
            echo -Xclang -target-feature -Xclang +ptx50
            ;;
        sm_*) ;;
    esac
}

# Populate array CUMETAL_CUDA_DEVICE_FLAGS with device-side compile arguments.
cumetal_cuda_device_flags() {
    local arch
    arch="$(cumetal_cuda_arch)"
    CUMETAL_CUDA_DEVICE_FLAGS=(
        --cuda-gpu-arch="${arch}"
    )
    local ptx_flags
    ptx_flags="$(cumetal_cuda_ptx_feature_flags "${arch}")"
    if [[ -n "${ptx_flags}" ]]; then
        # shellcheck disable=SC2206
        CUMETAL_CUDA_DEVICE_FLAGS+=(${ptx_flags})
    fi
}
