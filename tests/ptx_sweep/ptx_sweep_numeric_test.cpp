// Per-opcode numerical PTX sweep (spec.md §10.2).
//
// The pre-existing sweep (run_supported_ops.sh) lowers a minimal kernel per opcode and greps the
// emitted IR for `define void @name`. That proves the lowering did not error and emitted a
// function; it says nothing about what the function computes. Every silent-wrong-answer bug in
// docs/correctness-audit-2026-07-26.md would have passed it -- the broken `cvt.rni.f32.f32`
// lowered fine, it just returned an integer truncation instead of a rounded float.
//
// This harness instead *executes* each opcode on the Apple GPU and compares the result bit-for-bit
// against a hand-derived expected value.
//
// On the oracle: expected values are written here as literals derived by hand from the PTX ISA
// specification, not computed by any code path CuMetal shares. That is what makes this a test
// rather than a tautology. Every float case uses values exactly representable in binary32, so
// "bit-for-bit" is a fair requirement and there is no tolerance to tune. spec §10.2 also asks for
// comparison against a reference CUDA implementation on NVIDIA hardware; that requires hardware
// this project does not have, and a hand-derived ISA oracle is the strongest available substitute.
//
// Cases are classified SUPPORTED / WRONG / UNSUPPORTED, following
// functional_cuda_projects_libdevice_math. WRONG is always a failure. UNSUPPORTED is a failure
// too unless --allow-missing is passed, so coverage cannot quietly regress.

#include "cuda_runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

// Bit-exact conversions. These only encode a value; they never compute an expected result.
std::uint32_t bits_of(float value) {
    std::uint32_t out = 0;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

float float_of(std::uint32_t value) {
    float out = 0.0f;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

enum class Domain { kInteger, kFloat };

struct Case {
    const char* name;
    Domain domain;
    // PTX computing %r3 (integer) or %f3 (float) from %r1/%r2/%r4 or %f1/%f2/%f4.
    const char* body;
    std::uint32_t input0;
    std::uint32_t input1;
    std::uint32_t input2;
    std::uint32_t expected;
    // Why this expected value is what the PTX ISA requires. Printed on failure so a mismatch is
    // debuggable without re-deriving the semantics.
    const char* rationale;
};

// clang-format off
const Case kCases[] = {
    // ---- integer arithmetic ----
    {"add_s32", Domain::kInteger, "add.s32 %r3, %r1, %r2;",
     7u, 5u, 0u, 12u, "7 + 5"},
    {"sub_s32", Domain::kInteger, "sub.s32 %r3, %r1, %r2;",
     7u, 5u, 0u, 2u, "7 - 5"},
    {"mul_lo_s32", Domain::kInteger, "mul.lo.s32 %r3, %r1, %r2;",
     0x00010001u, 0x00010001u, 0u, 0x00020001u,
     "65537*65537 = 0x1_00020001; mul.lo keeps the low 32 bits"},
    {"mul_hi_s32", Domain::kInteger, "mul.hi.s32 %r3, %r1, %r2;",
     0x00010001u, 0x00010001u, 0u, 0x00000001u,
     "65537*65537 = 0x1_00020001; mul.hi keeps the high 32 bits"},
    {"mad_lo_s32", Domain::kInteger, "mad.lo.s32 %r3, %r1, %r2, %r4;",
     6u, 7u, 5u, 47u, "6*7 + 5"},
    {"div_s32_negative", Domain::kInteger, "div.s32 %r3, %r1, %r2;",
     0xFFFFFFF9u, 2u, 0u, 0xFFFFFFFDu,
     "-7 / 2 truncates toward zero to -3, not floor(-3.5) = -4"},
    {"rem_s32_negative", Domain::kInteger, "rem.s32 %r3, %r1, %r2;",
     0xFFFFFFF9u, 2u, 0u, 0xFFFFFFFFu,
     "-7 %% 2 takes the sign of the dividend: -1"},
    {"abs_s32", Domain::kInteger, "abs.s32 %r3, %r1;",
     0xFFFFFFF9u, 0u, 0u, 7u, "abs(-7)"},
    {"neg_s32", Domain::kInteger, "neg.s32 %r3, %r1;",
     7u, 0u, 0u, 0xFFFFFFF9u, "-(7)"},
    {"min_s32", Domain::kInteger, "min.s32 %r3, %r1, %r2;",
     0xFFFFFFF9u, 5u, 0u, 0xFFFFFFF9u, "signed min(-7, 5) = -7"},
    {"max_u32", Domain::kInteger, "max.u32 %r3, %r1, %r2;",
     0xFFFFFFF9u, 5u, 0u, 0xFFFFFFF9u,
     "unsigned max(4294967289, 5): the same bits compare the other way from min.s32"},

    // ---- bitwise and shifts ----
    {"and_b32", Domain::kInteger, "and.b32 %r3, %r1, %r2;",
     0xF0F0F0F0u, 0x0FF00FF0u, 0u, 0x00F000F0u, "bitwise and"},
    {"or_b32", Domain::kInteger, "or.b32 %r3, %r1, %r2;",
     0xF0F0F0F0u, 0x0FF00FF0u, 0u, 0xFFF0FFF0u, "bitwise or"},
    {"xor_b32", Domain::kInteger, "xor.b32 %r3, %r1, %r2;",
     0xF0F0F0F0u, 0x0FF00FF0u, 0u, 0xFF00FF00u, "bitwise xor"},
    {"not_b32", Domain::kInteger, "not.b32 %r3, %r1;",
     0xF0F0F0F0u, 0u, 0u, 0x0F0F0F0Fu, "bitwise complement"},
    {"shl_b32", Domain::kInteger, "shl.b32 %r3, %r1, %r2;",
     0x00000001u, 5u, 0u, 0x00000020u, "1 << 5"},
    {"shr_s32_arithmetic", Domain::kInteger, "shr.s32 %r3, %r1, %r2;",
     0x80000000u, 4u, 0u, 0xF8000000u, "arithmetic shift right sign-extends"},
    {"shr_u32_logical", Domain::kInteger, "shr.u32 %r3, %r1, %r2;",
     0x80000000u, 4u, 0u, 0x08000000u, "logical shift right zero-fills"},
    {"popc_b32", Domain::kInteger, "popc.b32 %r3, %r1;",
     0xF0F0F0F0u, 0u, 0u, 16u, "0xF0F0F0F0 has 16 set bits"},
    {"clz_b32", Domain::kInteger, "clz.b32 %r3, %r1;",
     0x00F00000u, 0u, 0u, 8u, "8 leading zeros before the top set bit"},
    {"brev_b32", Domain::kInteger, "brev.b32 %r3, %r1;",
     0x00000001u, 0u, 0u, 0x80000000u, "bit reversal moves bit 0 to bit 31"},

    // ---- float arithmetic; every value is exact in binary32 ----
    {"add_f32", Domain::kFloat, "add.f32 %f3, %f1, %f2;",
     bits_of(1.5f), bits_of(2.25f), 0u, bits_of(3.75f), "1.5 + 2.25"},
    {"sub_f32", Domain::kFloat, "sub.f32 %f3, %f1, %f2;",
     bits_of(1.5f), bits_of(2.25f), 0u, bits_of(-0.75f), "1.5 - 2.25"},
    {"mul_f32", Domain::kFloat, "mul.f32 %f3, %f1, %f2;",
     bits_of(1.5f), bits_of(2.25f), 0u, bits_of(3.375f), "1.5 * 2.25"},
    {"div_rn_f32", Domain::kFloat, "div.rn.f32 %f3, %f1, %f2;",
     bits_of(1.5f), bits_of(2.0f), 0u, bits_of(0.75f), "1.5 / 2"},
    {"fma_rn_f32", Domain::kFloat, "fma.rn.f32 %f3, %f1, %f2, %f4;",
     bits_of(1.5f), bits_of(2.25f), bits_of(0.5f), bits_of(3.875f), "1.5*2.25 + 0.5"},
    {"neg_f32", Domain::kFloat, "neg.f32 %f3, %f1;",
     bits_of(1.5f), 0u, 0u, bits_of(-1.5f), "negation"},
    {"abs_f32", Domain::kFloat, "abs.f32 %f3, %f1;",
     bits_of(-1.5f), 0u, 0u, bits_of(1.5f), "absolute value"},
    {"min_f32", Domain::kFloat, "min.f32 %f3, %f1, %f2;",
     bits_of(-1.5f), bits_of(2.25f), 0u, bits_of(-1.5f), "min(-1.5, 2.25)"},
    {"max_f32", Domain::kFloat, "max.f32 %f3, %f1, %f2;",
     bits_of(-1.5f), bits_of(2.25f), 0u, bits_of(2.25f), "max(-1.5, 2.25)"},
    {"sqrt_rn_f32", Domain::kFloat, "sqrt.rn.f32 %f3, %f1;",
     bits_of(2.25f), 0u, 0u, bits_of(1.5f), "sqrt(2.25) is exactly 1.5"},
    {"rcp_rn_f32", Domain::kFloat, "rcp.rn.f32 %f3, %f1;",
     bits_of(2.0f), 0u, 0u, bits_of(0.5f), "1/2 is exact"},

    // ---- cvt rounding modes ----
    // Direct regression coverage for the silent-wrong-answer bug fixed in fbaece1, where the
    // rounding mode was dropped on both lowering paths: cvt.rni.f32.f32 emitted an integer
    // truncation, so rintf(2.7) returned 2 and rintf(-1.5) clamped to 0.
    {"cvt_rni_f32_f32", Domain::kFloat, "cvt.rni.f32.f32 %f3, %f1;",
     bits_of(2.7f), 0u, 0u, bits_of(3.0f), "round to nearest integer: 2.7 -> 3.0"},
    {"cvt_rni_f32_f32_halfway", Domain::kFloat, "cvt.rni.f32.f32 %f3, %f1;",
     bits_of(2.5f), 0u, 0u, bits_of(2.0f),
     "round to nearest EVEN on a tie: 2.5 -> 2.0, not 3.0"},
    {"cvt_rni_f32_f32_negative", Domain::kFloat, "cvt.rni.f32.f32 %f3, %f1;",
     bits_of(-1.5f), 0u, 0u, bits_of(-2.0f),
     "round to nearest even: -1.5 -> -2.0; the old bug returned 0"},
    {"cvt_rzi_f32_f32", Domain::kFloat, "cvt.rzi.f32.f32 %f3, %f1;",
     bits_of(2.7f), 0u, 0u, bits_of(2.0f), "truncate toward zero"},
    {"cvt_rzi_f32_f32_negative", Domain::kFloat, "cvt.rzi.f32.f32 %f3, %f1;",
     bits_of(-2.7f), 0u, 0u, bits_of(-2.0f), "truncate toward zero, not toward -inf"},
    {"cvt_rmi_f32_f32", Domain::kFloat, "cvt.rmi.f32.f32 %f3, %f1;",
     bits_of(-1.5f), 0u, 0u, bits_of(-2.0f), "round toward -inf (floor)"},
    {"cvt_rpi_f32_f32", Domain::kFloat, "cvt.rpi.f32.f32 %f3, %f1;",
     bits_of(-1.5f), 0u, 0u, bits_of(-1.0f), "round toward +inf (ceil)"},

    // ---- cvt between domains ----
    {"cvt_rn_f32_s32", Domain::kFloat, "cvt.rn.f32.s32 %f3, %r1;",
     0xFFFFFFF9u, 0u, 0u, bits_of(-7.0f), "signed -7 to float"},
    {"cvt_rzi_s32_f32", Domain::kInteger, "cvt.rzi.s32.f32 %r3, %f1;",
     bits_of(2.7f), 0u, 0u, 2u, "float to signed int truncating toward zero"},
    {"cvt_rzi_s32_f32_negative", Domain::kInteger, "cvt.rzi.s32.f32 %r3, %f1;",
     bits_of(-2.7f), 0u, 0u, 0xFFFFFFFEu, "-2.7 truncates to -2"},
};
// clang-format on

constexpr std::size_t kCaseCount = sizeof(kCases) / sizeof(kCases[0]);

// Both templates load three input words and store one result word, so the host side is identical
// across cases. Registers are declared generously; unused ones cost nothing.
std::string build_ptx(const Case& test_case) {
    std::string ptx =
        ".version 8.0\n"
        ".target sm_80\n"
        ".address_size 64\n"
        ".visible .entry ";
    ptx += test_case.name;
    ptx +=
        "(\n"
        "  .param .u64 in,\n"
        "  .param .u64 out\n"
        ")\n"
        "{\n"
        "  .reg .pred %p<4>;\n"
        "  .reg .b64 %rd<8>;\n"
        "  .reg .b32 %r<8>;\n"
        "  .reg .f32 %f<8>;\n"
        "  ld.param.u64 %rd1, [in];\n"
        "  ld.param.u64 %rd2, [out];\n"
        "  cvta.to.global.u64 %rd3, %rd1;\n"
        "  cvta.to.global.u64 %rd4, %rd2;\n"
        // Load each input into both an integer and a float register so a case body can use
        // whichever it needs -- notably the cvt cases, which cross domains.
        "  ld.global.u32 %r1, [%rd3];\n"
        "  ld.global.u32 %r2, [%rd3+4];\n"
        "  ld.global.u32 %r4, [%rd3+8];\n"
        "  ld.global.f32 %f1, [%rd3];\n"
        "  ld.global.f32 %f2, [%rd3+4];\n"
        "  ld.global.f32 %f4, [%rd3+8];\n"
        "  ";
    ptx += test_case.body;
    ptx += "\n";
    ptx += (test_case.domain == Domain::kFloat) ? "  st.global.f32 [%rd4], %f3;\n"
                                                : "  st.global.u32 [%rd4], %r3;\n";
    ptx +=
        "  ret;\n"
        "}\n";
    return ptx;
}

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(c);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

enum class Result { kSupported, kWrong, kUnsupported };

struct Outcome {
    Result result = Result::kUnsupported;
    std::uint32_t actual = 0;
    std::string detail;
};

Outcome run_case(const Case& test_case,
                 const std::string& cumetalc,
                 const std::filesystem::path& workdir) {
    Outcome outcome;

    const std::filesystem::path ptx_path = workdir / (std::string(test_case.name) + ".ptx");
    const std::filesystem::path metallib_path =
        workdir / (std::string(test_case.name) + ".metallib");
    const std::filesystem::path log_path = workdir / (std::string(test_case.name) + ".log");

    {
        const std::string ptx = build_ptx(test_case);
        FILE* file = std::fopen(ptx_path.c_str(), "w");
        if (file == nullptr) {
            outcome.detail = "could not write PTX";
            return outcome;
        }
        std::fwrite(ptx.data(), 1, ptx.size(), file);
        std::fclose(file);
    }

    std::error_code ec;
    std::filesystem::remove(metallib_path, ec);

    const std::string command = shell_quote(cumetalc) + " --mode xcrun --overwrite --ptx-strict" +
                                " --entry " + shell_quote(test_case.name) + " --input " +
                                shell_quote(ptx_path.string()) + " --output " +
                                shell_quote(metallib_path.string()) + " > " +
                                shell_quote(log_path.string()) + " 2>&1";
    const int status = std::system(command.c_str());

    // A lowering that refuses is a coverage gap, not a wrong answer: report it as UNSUPPORTED so
    // the two never get conflated.
    if (status != 0 || !std::filesystem::exists(metallib_path)) {
        outcome.result = Result::kUnsupported;
        outcome.detail = "cumetalc declined to lower this opcode (see " + log_path.string() + ")";
        return outcome;
    }

    void* device_in = nullptr;
    void* device_out = nullptr;
    if (cudaMalloc(&device_in, 3 * sizeof(std::uint32_t)) != cudaSuccess ||
        cudaMalloc(&device_out, sizeof(std::uint32_t)) != cudaSuccess) {
        outcome.detail = "cudaMalloc failed";
        return outcome;
    }

    const std::uint32_t inputs[3] = {test_case.input0, test_case.input1, test_case.input2};
    // Poison the output so a kernel that never writes is caught rather than reading as a zero
    // that happens to match an expected zero.
    const std::uint32_t poison = 0xDEADBEEFu;
    if (cudaMemcpy(device_in, inputs, sizeof(inputs), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_out, &poison, sizeof(poison), cudaMemcpyHostToDevice) != cudaSuccess) {
        outcome.detail = "cudaMemcpy host->device failed";
        return outcome;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
    };
    const std::string metallib_string = metallib_path.string();
    const cumetalKernel_t kernel{
        .metallib_path = metallib_string.c_str(),
        .kernel_name = test_case.name,
        .arg_count = 2,
        .arg_info = kArgInfo,
    };

    void* arg_in = device_in;
    void* arg_out = device_out;
    void* launch_args[] = {&arg_in, &arg_out};

    if (cudaLaunchKernel(&kernel, dim3(1, 1, 1), dim3(1, 1, 1), launch_args, 0, nullptr) !=
        cudaSuccess) {
        outcome.result = Result::kUnsupported;
        outcome.detail = "kernel launch failed";
        cudaFree(device_in);
        cudaFree(device_out);
        return outcome;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        outcome.detail = "cudaDeviceSynchronize failed";
        cudaFree(device_in);
        cudaFree(device_out);
        return outcome;
    }

    std::uint32_t actual = 0;
    if (cudaMemcpy(&actual, device_out, sizeof(actual), cudaMemcpyDeviceToHost) != cudaSuccess) {
        outcome.detail = "cudaMemcpy device->host failed";
        cudaFree(device_in);
        cudaFree(device_out);
        return outcome;
    }

    cudaFree(device_in);
    cudaFree(device_out);

    outcome.actual = actual;
    if (actual == poison) {
        outcome.result = Result::kWrong;
        outcome.detail = "kernel did not write its output";
        return outcome;
    }
    outcome.result = (actual == test_case.expected) ? Result::kSupported : Result::kWrong;
    return outcome;
}

void print_value(const Case& test_case, std::uint32_t value, char* buffer, std::size_t size) {
    if (test_case.domain == Domain::kFloat) {
        std::snprintf(buffer, size, "0x%08X (%g)", value,
                      static_cast<double>(float_of(value)));
    } else {
        std::snprintf(buffer, size, "0x%08X (%d)", value, static_cast<std::int32_t>(value));
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <cumetalc> <workdir> [--allow-missing]\n", argv[0]);
        return 64;
    }

    const std::string cumetalc = argv[1];
    const std::filesystem::path workdir = argv[2];
    bool allow_missing = false;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--allow-missing") == 0) allow_missing = true;
    }

    if (!std::filesystem::exists(cumetalc)) {
        std::fprintf(stderr, "SKIP: cumetalc not found at %s\n", cumetalc.c_str());
        return 77;
    }

    std::error_code ec;
    std::filesystem::create_directories(workdir, ec);

    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "SKIP: no CUDA/Metal device available\n");
        return 77;
    }

    std::size_t supported = 0;
    std::size_t wrong = 0;
    std::size_t unsupported = 0;
    std::vector<std::string> wrong_names;
    std::vector<std::string> unsupported_names;

    std::printf("%-28s %-12s %s\n", "OPCODE CASE", "RESULT", "DETAIL");
    for (std::size_t i = 0; i < kCaseCount; ++i) {
        const Case& test_case = kCases[i];
        const Outcome outcome = run_case(test_case, cumetalc, workdir);

        char expected_text[64];
        char actual_text[64];
        print_value(test_case, test_case.expected, expected_text, sizeof(expected_text));
        print_value(test_case, outcome.actual, actual_text, sizeof(actual_text));

        switch (outcome.result) {
            case Result::kSupported:
                ++supported;
                std::printf("%-28s %-12s %s\n", test_case.name, "SUPPORTED", expected_text);
                break;
            case Result::kWrong:
                ++wrong;
                wrong_names.emplace_back(test_case.name);
                std::printf("%-28s %-12s got %s, expected %s -- %s\n",
                            test_case.name,
                            "WRONG",
                            actual_text,
                            expected_text,
                            test_case.rationale);
                if (!outcome.detail.empty()) {
                    std::printf("%-28s %-12s %s\n", "", "", outcome.detail.c_str());
                }
                break;
            case Result::kUnsupported:
                ++unsupported;
                unsupported_names.emplace_back(test_case.name);
                std::printf("%-28s %-12s %s\n",
                            test_case.name,
                            "UNSUPPORTED",
                            outcome.detail.c_str());
                break;
        }
    }

    std::printf("\n%zu cases: %zu supported, %zu wrong, %zu unsupported\n",
                kCaseCount, supported, wrong, unsupported);

    if (wrong > 0) {
        std::printf("\nFAIL: %zu opcode(s) computed the wrong value:\n", wrong);
        for (const std::string& name : wrong_names) {
            std::printf("  - %s\n", name.c_str());
        }
        std::printf(
            "A wrong value is never a coverage gap. Either the lowering is broken or the expected\n"
            "value in this file misreads the PTX ISA; both need a human, not a relaxed tolerance.\n");
        return 1;
    }

    if (unsupported > 0 && !allow_missing) {
        std::printf("\nFAIL: %zu opcode(s) could not be lowered:\n", unsupported);
        for (const std::string& name : unsupported_names) {
            std::printf("  - %s\n", name.c_str());
        }
        std::printf("Pass --allow-missing to treat these as acceptable coverage gaps.\n");
        return 1;
    }

    std::printf("\nPASS: every covered PTX opcode computed the ISA-specified value\n");
    return 0;
}
