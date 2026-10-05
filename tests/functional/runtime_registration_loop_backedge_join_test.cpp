// A loop whose backedge sits in both arms of a branch that rejoins only at the
// loop exit.
//
// The loop header branches into two in-loop paths that join (`$JOIN`). The
// join then splits into a "found" arm and a "missing" arm that may leave the
// loop; both arms reach the latch, which either takes the backedge or exits.
// The arms' nearest common successor is therefore the loop exit. The typed
// MSL structurizer emitted the latch inside each arm, assigned the header's
// arguments for the backedge and then fell through into the join's `break`,
// so the loop ran exactly once and every launch reported success.
//
// This is the CFG clang produces for LAMMPS' NeighBondKokkos::bond_all (map
// style select in the header, `lost/bond error` early return, atomic append):
// only each atom's first bond was listed, so GPU bond and angle energies were
// wrong whenever the neighbour build ran on the device.
#include "cuda_runtime.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
void** __cudaRegisterFatBinary(const void* fat_cubin);
void __cudaUnregisterFatBinary(void** fat_cubin_handle);
void __cudaRegisterFunction(void** fat_cubin_handle,
                            const void* host_function,
                            char* device_function,
                            const char* device_name,
                            int thread_limit,
                            void* thread_id,
                            void* block_id,
                            void* block_dim,
                            void* grid_dim,
                            int* warp_size);
}

namespace {

constexpr std::uint32_t kFatbinWrapperMagic = 0x466243b1u;
constexpr std::uint32_t kFatbinBlobMagic = 0xBA55ED50u;
constexpr int kThreads = 64;
constexpr int kMissingWeight = 1000;

struct FatbinWrapper {
    std::uint32_t magic = kFatbinWrapperMagic;
    std::uint32_t version = 1;
    const void* data = nullptr;
    const void* unknown = nullptr;
};

struct FatbinBlobHeader {
    std::uint32_t magic = kFatbinBlobMagic;
    std::uint16_t version = 1;
    std::uint16_t header_size = 16;
    std::uint64_t fat_size = 0;
};

// out[i] = sum over m < counts[i] of (m + 1), where under style 2 the m == 2
// lookup "misses": it adds 1000 instead and, with stop set, ends the walk.
const char kPtx[] = R"PTX(
.version 7.0
.target sm_80
.address_size 64

.visible .entry loop_backedge_join(
    .param .u64 loop_backedge_join_param_0,
    .param .u64 loop_backedge_join_param_1,
    .param .u32 loop_backedge_join_param_2,
    .param .u32 loop_backedge_join_param_3)
{
    .reg .pred %p<8>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [loop_backedge_join_param_0];
    ld.param.u64 %rd2, [loop_backedge_join_param_1];
    ld.param.u32 %r1, [loop_backedge_join_param_2];
    ld.param.u32 %r2, [loop_backedge_join_param_3];
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    mov.u32 %r3, %tid.x;
    mul.wide.u32 %rd5, %r3, 4;
    add.s64 %rd6, %rd3, %rd5;
    add.s64 %rd7, %rd4, %rd5;
    ld.global.u32 %r4, [%rd6];
    setp.eq.s32 %p1, %r1, 2;
    setp.ne.s32 %p5, %r2, 0;
    mov.b32 %r10, 0;
    mov.b32 %r11, 0;
    setp.lt.s32 %p2, %r4, 1;
    @%p2 bra $EXIT;
$HEAD:
    @%p1 bra $HASH;
    add.s32 %r12, %r10, 1;
    bra.uni $JOIN;
$HASH:
    setp.eq.s32 %p3, %r10, 2;
    add.s32 %r13, %r10, 1;
    selp.b32 %r12, -1, %r13, %p3;
$JOIN:
    setp.ne.s32 %p4, %r12, -1;
    @!%p4 bra $MISS;
    add.s32 %r11, %r11, %r12;
    bra.uni $LATCH;
$MISS:
    add.s32 %r11, %r11, 1000;
    @%p5 bra $EXIT;
$LATCH:
    add.s32 %r10, %r10, 1;
    setp.lt.s32 %p6, %r10, %r4;
    @%p6 bra $HEAD;
$EXIT:
    st.global.u32 [%rd7], %r11;
    ret;
}
)PTX";

void loop_backedge_join_host_stub() {}

int expected(int count, int style, int stop) {
    int sum = 0;
    for (int m = 0; m < count; ++m) {
        if (style == 2 && m == 2) {
            sum += kMissingWeight;
            if (stop) break;
            continue;
        }
        sum += m + 1;
    }
    return sum;
}

}  // namespace

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    const std::string ptx(kPtx);
    std::vector<std::uint8_t> fatbin_blob(sizeof(FatbinBlobHeader) + ptx.size() + 1, 0);
    FatbinBlobHeader header{};
    header.fat_size = static_cast<std::uint64_t>(ptx.size() + 1);
    std::memcpy(fatbin_blob.data(), &header, sizeof(header));
    std::memcpy(fatbin_blob.data() + sizeof(header), ptx.data(), ptx.size());

    FatbinWrapper wrapper{};
    wrapper.data = fatbin_blob.data();
    void** fatbin_handle = __cudaRegisterFatBinary(&wrapper);
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: __cudaRegisterFatBinary returned null\n");
        return 1;
    }
    char device_function[] = "loop_backedge_join";
    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&loop_backedge_join_host_stub),
                           device_function, nullptr, 0,
                           nullptr, nullptr, nullptr, nullptr, nullptr);

    int counts[kThreads];
    for (int i = 0; i < kThreads; ++i) counts[i] = i % 7;  // 0..6 trips
    void* d_counts = nullptr;
    void* d_out = nullptr;
    if (cudaMalloc(&d_counts, sizeof counts) != cudaSuccess ||
        cudaMalloc(&d_out, sizeof counts) != cudaSuccess ||
        cudaMemcpy(d_counts, counts, sizeof counts, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: device setup failed\n");
        return 1;
    }

    int failures = 0;
    for (int style = 1; style <= 2; ++style) {
        for (int stop = 0; stop <= 1; ++stop) {
            if (cudaMemset(d_out, 0xff, sizeof counts) != cudaSuccess) return 1;
            void* arg_counts = d_counts;
            void* arg_out = d_out;
            int arg_style = style;
            int arg_stop = stop;
            void* args[] = {&arg_counts, &arg_out, &arg_style, &arg_stop};
            if (cudaLaunchKernel(reinterpret_cast<const void*>(&loop_backedge_join_host_stub),
                                 dim3(1), dim3(kThreads), args, 0, nullptr) != cudaSuccess ||
                cudaDeviceSynchronize() != cudaSuccess) {
                std::fprintf(stderr, "FAIL: launch failed (style=%d stop=%d)\n", style, stop);
                return 1;
            }
            int out[kThreads];
            if (cudaMemcpy(out, d_out, sizeof out, cudaMemcpyDeviceToHost) != cudaSuccess) {
                std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
                return 1;
            }
            for (int i = 0; i < kThreads; ++i) {
                const int want = expected(counts[i], style, stop);
                if (out[i] != want) {
                    if (failures < 8) {
                        std::fprintf(stderr,
                                     "FAIL: style=%d stop=%d thread %d (%d trips): got %d, "
                                     "expected %d\n",
                                     style, stop, i, counts[i], out[i], want);
                    }
                    ++failures;
                }
            }
        }
    }

    cudaFree(d_counts);
    cudaFree(d_out);
    __cudaUnregisterFatBinary(fatbin_handle);
    if (failures) {
        std::fprintf(stderr, "FAIL: %d mismatches\n", failures);
        return 1;
    }
    std::printf("LOOP_BACKEDGE_JOIN_OK\n");
    return 0;
}
