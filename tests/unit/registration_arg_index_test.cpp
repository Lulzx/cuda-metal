#include "registration.h"

#include <chrono>
#include <cstdio>
#include <string>

namespace {

bool check_arg(const std::vector<cumetalKernelArgInfo_t>& args,
               std::size_t index,
               int kind,
               std::uint32_t size) {
    return index < args.size() && args[index].kind == kind && args[index].size_bytes == size;
}

}  // namespace

int main() {
    const std::string fixture = R"ptx(
.version 8.0
.target sm_80
.address_size 64
// .entry comment_only(.param .u64 wrong)
.file 1 ".entry string_only(.param .u64 wrong)"
/*
.entry block_comment_only(.param .u64 wrong)
*/
.visible .entry pointer_scalar(
    .param .u64 .ptr .global .align 16 pointer_scalar_param_0,
    .param .u32 pointer_scalar_param_1,
    /* a by-value aggregate */
    .param .align 8 .b8 pointer_scalar_param_2[24]
)
.maxntid 256, 1, 1
{
    ret;
}
.extern .entry no_args()
{
    ret;
}
.visible .entry legacy_unqualified(
    .param .u64 legacy_pointer,
    .param .f64 scalar_double,
    .param .f16 tolerant_half
)
{
    ret;
}
)ptx";

    const auto index = cumetal::registration::build_arg_info_index_from_ptx(fixture);
    const auto kernel = index.find("pointer_scalar");
    const auto empty = index.find("no_args");
    const auto legacy = index.find("legacy_unqualified");
    if (index.size() != 3 || kernel == index.end() || empty == index.end() ||
        legacy == index.end() ||
        kernel->second.size() != 3 || !empty->second.empty() ||
        legacy->second.size() != 3 ||
        !check_arg(kernel->second, 0, CUMETAL_ARG_BUFFER, sizeof(void*)) ||
        !check_arg(kernel->second, 1, CUMETAL_ARG_BYTES, 4) ||
        !check_arg(kernel->second, 2, CUMETAL_ARG_BYTES, 24) ||
        !check_arg(legacy->second, 0, CUMETAL_ARG_BUFFER, sizeof(void*)) ||
        !check_arg(legacy->second, 1, CUMETAL_ARG_BYTES, 8) ||
        !check_arg(legacy->second, 2, CUMETAL_ARG_BYTES, 4)) {
        std::fprintf(stderr, "FAIL: PTX entry signature ABI scan mismatch\n");
        return 1;
    }

    std::vector<cumetalKernelArgInfo_t> selected;
    if (!cumetal::registration::find_arg_info_for_ptx_entry(
            fixture, "legacy_unqualified", &selected) ||
        selected.size() != 3 ||
        !check_arg(selected, 0, CUMETAL_ARG_BUFFER, sizeof(void*)) ||
        !check_arg(selected, 1, CUMETAL_ARG_BYTES, 8) ||
        cumetal::registration::find_arg_info_for_ptx_entry(
            fixture, "comment_only", &selected) ||
        cumetal::registration::find_arg_info_for_ptx_entry(
            fixture, "string_only", &selected) ||
        cumetal::registration::find_arg_info_for_ptx_entry(
            fixture, "block_comment_only", &selected) ||
        cumetal::registration::find_arg_info_for_ptx_entry(
            fixture, "missing", &selected)) {
        std::fprintf(stderr, "FAIL: targeted PTX entry ABI scan mismatch\n");
        return 1;
    }

    std::string large;
    constexpr int kEntries = 5000;
    large.reserve(static_cast<std::size_t>(kEntries) * 160);
    large.append(".version 8.0\n.target sm_80\n.address_size 64\n");
    for (int i = 0; i < kEntries; ++i) {
        large.append(".visible .entry kernel_");
        large.append(std::to_string(i));
        large.append("(.param .u64 .ptr .global p, .param .u32 n) { ret; }\n");
    }

    const auto start = std::chrono::steady_clock::now();
    const auto large_index = cumetal::registration::build_arg_info_index_from_ptx(large);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    if (large_index.size() != kEntries || elapsed > std::chrono::seconds(2)) {
        std::fprintf(stderr,
                     "FAIL: linear PTX ABI scan produced %zu/%d entries in %lld ms\n",
                     large_index.size(),
                     kEntries,
                     static_cast<long long>(elapsed.count()));
        return 1;
    }

    const auto targeted_start = std::chrono::steady_clock::now();
    if (!cumetal::registration::find_arg_info_for_ptx_entry(
            large, "kernel_4999", &selected) ||
        selected.size() != 2 ||
        !check_arg(selected, 0, CUMETAL_ARG_BUFFER, sizeof(void*)) ||
        !check_arg(selected, 1, CUMETAL_ARG_BYTES, 4)) {
        std::fprintf(stderr, "FAIL: targeted large PTX ABI scan mismatch\n");
        return 1;
    }
    const auto targeted_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - targeted_start);

    std::printf("PASS: scanned %d PTX entry ABIs in %lld ms; targeted last entry in %lld ms\n",
                kEntries,
                static_cast<long long>(elapsed.count()),
                static_cast<long long>(targeted_elapsed.count()));
    return 0;
}
