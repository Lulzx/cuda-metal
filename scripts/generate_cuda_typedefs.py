#!/usr/bin/env python3
"""Generate runtime/api/cudaTypedefs.h from runtime/api/cuda.h.

NVIDIA's cudaTypedefs.h spells every driver entry point as a versioned function
pointer typedef (PFN_cuFoo_vNNNN). Hosts that load the driver dynamically, such
as NVIDIA Warp, declare their pointers with those names. CuMetal exports each
entry point once with the CUDA 12 ABI, so the typedef for every version is the
same signature: the version suffix records the ABI generation NVIDIA introduced
it in, and this generator takes the suffixes from a fixed table rather than
from the header so the names stay byte-identical to NVIDIA's.

Entry points CuMetal does not implement are typed here as well, because callers
still declare the pointer; cuGetProcAddress reports them as not found.

Usage: scripts/generate_cuda_typedefs.py [--check]
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CUDA_H = ROOT / "runtime" / "api" / "cuda.h"
OUT = ROOT / "runtime" / "api" / "cudaTypedefs.h"

# name -> version suffix, as spelled in NVIDIA's cudaTypedefs.h for the CUDA 12 ABI.
VERSIONS = {
    "cuArray3DCreate": 3020, "cuArrayCreate": 3020, "cuArrayDestroy": 2000,
    "cuCtxCreate": 3020, "cuCtxDestroy": 4000, "cuCtxDisablePeerAccess": 4000,
    "cuCtxEnablePeerAccess": 4000, "cuCtxGetCurrent": 4000, "cuCtxGetDevice": 2000,
    "cuCtxPopCurrent": 4000, "cuCtxPushCurrent": 4000, "cuCtxSetCurrent": 4000,
    "cuCtxSynchronize": 2000, "cuDeviceCanAccessPeer": 4000, "cuDeviceGet": 2000,
    "cuDeviceGetAttribute": 2000, "cuDeviceGetCount": 2000, "cuDeviceGetName": 2000,
    "cuDeviceGetUuid": 11040, "cuDevicePrimaryCtxRelease": 11000,
    "cuDevicePrimaryCtxRetain": 7000, "cuDriverGetVersion": 2020, "cuEventCreate": 2000,
    "cuEventDestroy": 4000, "cuEventQuery": 2000, "cuEventRecord": 2000,
    "cuEventRecordWithFlags": 11010, "cuEventSynchronize": 2000, "cuFuncSetAttribute": 9000,
    "cuGetErrorName": 6000, "cuGetErrorString": 6000, "cuGetProcAddress": 12000,
    "cuGraphAddNode": 12030, "cuGraphicsMapResources": 3000,
    "cuGraphicsResourceGetMappedPointer": 3020, "cuGraphicsUnmapResources": 3000,
    "cuGraphicsUnregisterResource": 3000, "cuGraphNodeGetDependentNodes": 10000,
    "cuGraphNodeGetType": 10000, "cuInit": 2000, "cuIpcCloseMemHandle": 4010,
    "cuIpcGetEventHandle": 4010, "cuIpcGetMemHandle": 4010, "cuIpcOpenEventHandle": 4010,
    "cuIpcOpenMemHandle": 11000, "cuLaunchKernel": 4000, "cuMemcpy2D": 3020,
    "cuMemcpy2DAsync": 3020, "cuMemcpy3D": 3020, "cuMemcpy3DAsync": 3020,
    "cuMemcpyBatchAsync": 12080, "cuMemcpyPeerAsync": 4000, "cuMemGetInfo": 3020,
    "cuModuleGetFunction": 2000, "cuModuleGetGlobal": 3020, "cuModuleLoadDataEx": 2010,
    "cuModuleUnload": 2000, "cuPointerGetAttribute": 4000, "cuStreamCreate": 2000,
    "cuStreamCreateWithPriority": 5050, "cuStreamDestroy": 4000,
    "cuStreamGetCaptureInfo": 11030, "cuStreamGetCtx": 9020, "cuStreamGetPriority": 5050,
    "cuStreamQuery": 2000, "cuStreamSynchronize": 2000,
    "cuStreamUpdateCaptureDependencies": 11030, "cuStreamWaitEvent": 3020,
    "cuTexObjectCreate": 5000, "cuTexObjectDestroy": 5000,
    "cuMemAlloc": 3020, "cuMemFree": 3020, "cuMemcpyHtoD": 3020, "cuMemcpyDtoH": 3020,
    "cuMemcpyDtoD": 3020, "cuMemcpyHtoDAsync": 3020, "cuMemcpyDtoHAsync": 3020,
    "cuMemcpyDtoDAsync": 3020, "cuMemsetD8": 3020, "cuMemsetD32": 3020,
    "cuMemsetD8Async": 3020, "cuMemsetD32Async": 3020, "cuStreamGetFlags": 5000,
    "cuMemAllocHost": 3020, "cuMemFreeHost": 2000, "cuMemAllocManaged": 6000,
    "cuModuleLoadData": 2000, "cuModuleLoad": 2000, "cuCtxGetFlags": 7000,
    "cuDeviceTotalMem": 3020, "cuDeviceComputeCapability": 2000,
    "cuMemHostAlloc": 2020, "cuMemHostGetDevicePointer": 3020,
    "cuStreamAddCallback": 5000, "cuLaunchHostFunc": 10000, "cuEventElapsedTime": 2000,
}

# Signatures for entry points cuda.h does not declare (not implemented).
UNDECLARED = {
    "cuGraphAddNode": "CUgraphNode* phGraphNode, CUgraph hGraph, const CUgraphNode* dependencies, "
                      "const CUgraphEdgeData* dependencyData, size_t numDependencies, "
                      "CUgraphNodeParams* nodeParams",
    "cuGraphicsMapResources": "unsigned int count, CUgraphicsResource* resources, CUstream hStream",
    "cuGraphicsResourceGetMappedPointer": "CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource",
    "cuGraphicsUnmapResources": "unsigned int count, CUgraphicsResource* resources, CUstream hStream",
    "cuGraphicsUnregisterResource": "CUgraphicsResource resource",
    "cuGraphNodeGetDependentNodes": "CUgraphNode hNode, CUgraphNode* dependentNodes, size_t* numDependentNodes",
    "cuGraphNodeGetType": "CUgraphNode hNode, CUgraphNodeType* type",
    "cuIpcCloseMemHandle": "CUdeviceptr dptr",
    "cuIpcGetEventHandle": "CUipcEventHandle* pHandle, CUevent event",
    "cuIpcGetMemHandle": "CUipcMemHandle* pHandle, CUdeviceptr dptr",
    "cuIpcOpenEventHandle": "CUevent* phEvent, CUipcEventHandle handle",
    "cuIpcOpenMemHandle": "CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int Flags",
    "cuStreamGetCaptureInfo": "CUstream hStream, CUstreamCaptureStatus* captureStatus_out, "
                              "cuuint64_t* id_out, CUgraph* graph_out, const CUgraphNode** dependencies_out, "
                              "size_t* numDependencies_out",
    "cuStreamUpdateCaptureDependencies": "CUstream hStream, CUgraphNode* dependencies, "
                                         "size_t numDependencies, unsigned int flags",
    "cuArrayCreate": "CUarray* pHandle, const CUDA_ARRAY_DESCRIPTOR* pAllocateArray",
}


def declared_signatures(text: str) -> dict[str, str]:
    """Return {name: parameter list} for every CUresult cuX(...) declaration."""
    signatures: dict[str, str] = {}
    for match in re.finditer(r"^CUresult\s+(cu[A-Za-z0-9_]+)\s*\((.*?)\)\s*;", text, re.S | re.M):
        params = " ".join(match.group(2).split())
        params = re.sub(r"/\*.*?\*/", "", params)
        params = re.sub(r"\s*,\s*", ", ", params).strip()
        signatures[match.group(1)] = params
    return signatures


def render() -> str:
    declared = declared_signatures(CUDA_H.read_text())
    lines = [
        "// Generated by scripts/generate_cuda_typedefs.py from runtime/api/cuda.h.",
        "// Do not edit by hand; rerun the generator after changing the driver API.",
        "//",
        "// Versioned driver entry-point typedefs in the shape of NVIDIA's",
        "// cudaTypedefs.h. CuMetal exports each entry point once with the CUDA 12",
        "// ABI, so every version of a name shares one signature.",
        "#ifndef CUMETAL_CUDA_TYPEDEFS_H",
        "#define CUMETAL_CUDA_TYPEDEFS_H",
        "",
        '#include "cuda.h"',
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
    ]
    missing = []
    for name in sorted(VERSIONS):
        params = declared.get(name) or UNDECLARED.get(name)
        if params is None:
            missing.append(name)
            continue
        status = "" if name in declared else "  // not implemented; cuGetProcAddress reports NOT_FOUND"
        lines.append(f"typedef CUresult (CUDAAPI *PFN_{name}_v{VERSIONS[name]})({params});{status}")
    if missing:
        raise SystemExit(f"no signature for: {', '.join(missing)}")
    lines += ["", "#ifdef __cplusplus", "}", "#endif", "", "#endif  // CUMETAL_CUDA_TYPEDEFS_H", ""]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    rendered = render()
    if "--check" in argv:
        current = OUT.read_text() if OUT.exists() else ""
        if current != rendered:
            print(f"{OUT} is stale; rerun scripts/generate_cuda_typedefs.py", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(rendered)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
