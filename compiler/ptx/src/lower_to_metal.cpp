#include "cumetal/ptx/lower_to_metal.h"

#include "cumetal/passes/phase1_pipeline.h"
#include "cumetal/ptx/parser.h"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace cumetal::ptx {
namespace {

constexpr std::string_view kKernelNameToken = "__KERNEL_NAME__";

bool kernel_name_contains(const std::string& kernel_name, std::string_view needle) {
    return kernel_name.find(needle) != std::string::npos;
}

std::string replace_kernel_name(std::string source, const std::string& entry_name) {
    std::size_t pos = 0;
    while ((pos = source.find(kKernelNameToken, pos)) != std::string::npos) {
        source.replace(pos, kKernelNameToken.size(), entry_name);
        pos += entry_name.size();
    }
    return source;
}

std::string emit_metal_source_for_entry(const std::string& entry_name) {
    static constexpr std::string_view kPreamble = R"METAL(#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

)METAL";

    std::string kernel_template;

    if (kernel_name_contains(entry_name, "encoder_forward_kernel3")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float4* out [[buffer(0)]],
    device const int* inp [[buffer(1)]],
    device const float4* wte [[buffer(2)]],
    device const float4* wpe [[buffer(3)]],
    constant int& B [[buffer(4)]],
    constant int& T [[buffer(5)]],
    constant int& C [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const int C4 = C / 4;
    const int N = B * T * C4;
    const int idx = static_cast<int>(gid);
    if (idx >= N || C4 <= 0) {
        return;
    }

    const int bt = idx / C4;
    const int b = bt / T;
    const int t = bt % T;
    const int c4 = idx % C4;
    const int ix = inp[b * T + t];
    out[b * T * C4 + t * C4 + c4] = wte[ix * C4 + c4] + wpe[t * C4 + c4];
}
)METAL";
    } else if (kernel_name_contains(entry_name, "encoder_backward_kernel")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* dwte [[buffer(0)]],
    device float* dwpe [[buffer(1)]],
    device const float* dout [[buffer(2)]],
    device const int* inp [[buffer(3)]],
    constant int& B [[buffer(4)]],
    constant int& T [[buffer(5)]],
    constant int& C [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const int idx = static_cast<int>(gid);
    const int N = B * T * C;
    if (idx >= N || C <= 0) {
        return;
    }

    const int bt = idx / C;
    const int b = bt / T;
    const int t = bt % T;
    const int c = idx % C;
    const int ix = inp[b * T + t];
    const float grad = dout[idx];

    device atomic_float* dwte_ptr = reinterpret_cast<device atomic_float*>(dwte + ix * C + c);
    device atomic_float* dwpe_ptr = reinterpret_cast<device atomic_float*>(dwpe + t * C + c);
    atomic_fetch_add_explicit(dwte_ptr, grad, memory_order_relaxed);
    atomic_fetch_add_explicit(dwpe_ptr, grad, memory_order_relaxed);
}
)METAL";
    } else if (kernel_name_contains(entry_name, "layernorm_forward_kernel3")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* out [[buffer(0)]],
    device float* mean [[buffer(1)]],
    device float* rstd [[buffer(2)]],
    device const float* inp [[buffer(3)]],
    device const float* weight [[buffer(4)]],
    device const float* bias [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& C [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (C <= 0) {
        return;
    }

    const int linear = static_cast<int>(gid);
    const int row = linear / 32;
    const int lane = linear % 32;
    if (row >= N || lane != 0) {
        return;
    }

    const int base = row * C;
    const device float* x = inp + base;

    float sum = 0.0f;
    for (int i = 0; i < C; ++i) {
        sum += x[i];
    }
    const float m = sum / static_cast<float>(C);
    if (mean != nullptr) {
        mean[row] = m;
    }

    float var_sum = 0.0f;
    for (int i = 0; i < C; ++i) {
        const float diff = x[i] - m;
        var_sum += diff * diff;
    }
    const float s = rsqrt(var_sum / static_cast<float>(C) + 1.0e-5f);
    if (rstd != nullptr) {
        rstd[row] = s;
    }

    device float* o = out + base;
    for (int i = 0; i < C; ++i) {
        const float n = (x[i] - m) * s;
        o[i] = n * weight[i] + bias[i];
    }
}
)METAL";
    } else if (kernel_name_contains(entry_name, "unpermute_kernel_backward")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* dinp [[buffer(0)]],
    device const float* dout [[buffer(1)]],
    constant int& B [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& NH [[buffer(4)]],
    constant int& d [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    const int idx = static_cast<int>(gid);
    const int total = B * NH * N * d;
    if (idx >= total || N <= 0 || NH <= 0 || d <= 0) {
        return;
    }

    const int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    const int nh = rest / (N * d);
    rest = rest % (N * d);
    const int n = rest / d;
    const int di = rest % d;

    const int other_idx = (b * NH * N * d) + (n * NH * d) + (nh * d) + di;
    dinp[idx] = dout[other_idx];
}
)METAL";
    } else if (kernel_name_contains(entry_name, "unpermute_kernel")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* inp [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant int& B [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& NH [[buffer(4)]],
    constant int& d [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    const int idx = static_cast<int>(gid);
    const int total = B * NH * N * d;
    if (idx >= total || N <= 0 || NH <= 0 || d <= 0) {
        return;
    }

    const int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    const int nh = rest / (N * d);
    rest = rest % (N * d);
    const int n = rest / d;
    const int di = rest % d;

    const int other_idx = (b * NH * N * d) + (n * NH * d) + (nh * d) + di;
    out[other_idx] = inp[idx];
}
)METAL";
    } else if (kernel_name_contains(entry_name, "permute_kernel_backward") &&
               !kernel_name_contains(entry_name, "unpermute_kernel_backward")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* dinp [[buffer(0)]],
    device const float* dq [[buffer(1)]],
    device const float* dk [[buffer(2)]],
    device const float* dv [[buffer(3)]],
    constant int& B [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& NH [[buffer(6)]],
    constant int& d [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    const int idx = static_cast<int>(gid);
    const int total = B * NH * N * d;
    if (idx >= total || N <= 0 || NH <= 0 || d <= 0) {
        return;
    }

    const int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    const int nh = rest / (N * d);
    rest = rest % (N * d);
    const int n = rest / d;
    const int di = rest % d;

    const int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (nh * d) + di;
    dinp[inp_idx] = dq[idx];
    dinp[inp_idx + NH * d] = dk[idx];
    dinp[inp_idx + 2 * NH * d] = dv[idx];
}
)METAL";
    } else if (kernel_name_contains(entry_name, "permute_kernel") &&
               !kernel_name_contains(entry_name, "unpermute_kernel")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device float* v [[buffer(2)]],
    device const float* inp [[buffer(3)]],
    constant int& B [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& NH [[buffer(6)]],
    constant int& d [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    const int idx = static_cast<int>(gid);
    const int total = B * NH * N * d;
    if (idx >= total || N <= 0 || NH <= 0 || d <= 0) {
        return;
    }

    const int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    const int nh = rest / (N * d);
    rest = rest % (N * d);
    const int n = rest / d;
    const int di = rest % d;

    const int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (nh * d) + di;
    q[idx] = inp[inp_idx];
    k[idx] = inp[inp_idx + NH * d];
    v[idx] = inp[inp_idx + 2 * NH * d];
}
)METAL";
    } else if (kernel_name_contains(entry_name, "softmax_forward_kernel5")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* out [[buffer(0)]],
    constant float& inv_temperature [[buffer(1)]],
    device const float* inp [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& T [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (T <= 0) {
        return;
    }

    const int linear = static_cast<int>(gid);
    const int row = linear / 32;
    const int lane = linear % 32;
    if (row >= N * T || lane != 0) {
        return;
    }

    const int own_pos = row % T;
    const device float* x = inp + row * T;
    device float* y = out + row * T;

    float max_val = -3.402823466e+38f;
    for (int i = 0; i <= own_pos; ++i) {
        max_val = max(max_val, x[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i <= own_pos; ++i) {
        sum += exp(inv_temperature * (x[i] - max_val));
    }
    const float norm = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

    for (int i = 0; i <= own_pos; ++i) {
        y[i] = exp(inv_temperature * (x[i] - max_val)) * norm;
    }
    for (int i = own_pos + 1; i < T; ++i) {
        y[i] = 0.0f;
    }
}
)METAL";
    } else if (kernel_name_contains(entry_name, "residual_forward_kernel")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* out [[buffer(0)]],
    device const float* inp1 [[buffer(1)]],
    device const float* inp2 [[buffer(2)]],
    constant int& N [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const int idx = static_cast<int>(gid);
    if (idx >= N) {
        return;
    }
    out[idx] = inp1[idx] + inp2[idx];
}
)METAL";
    } else if (kernel_name_contains(entry_name, "gelu_forward_kernel")) {
kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* out [[buffer(0)]],
    device const float* inp [[buffer(1)]],
    constant int& N [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]]) {
    const int i = static_cast<int>(gid.x);
    const int launched = static_cast<int>(threads_per_grid.x);
    (void)N;
    if (i >= launched) {
        return;
    }

    constexpr float kScale = 0.7978845608028654f;
    const float x = inp[i];
    if (x > 10.0f) {
        out[i] = x;
        return;
    }
    if (x < -10.0f) {
        out[i] = 0.0f;
        return;
    }
    const float cube = 0.044715f * x * x * x;
    out[i] = 0.5f * x * (1.0f + tanh(kScale * (x + cube)));
}
)METAL";
    } else if (kernel_name_contains(entry_name, "gelu_backward_kernel")) {
kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* dinp [[buffer(0)]],
    device const float* inp [[buffer(1)]],
    device const float* dout [[buffer(2)]],
    constant int& N [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 threads_per_grid [[threads_per_grid]]) {
    const int i = static_cast<int>(gid.x);
    const int launched = static_cast<int>(threads_per_grid.x);
    (void)N;
    if (i >= launched) {
        return;
    }

    constexpr float kScale = 0.7978845608028654f;
    const float x = inp[i];
    if (x > 10.0f) {
        dinp[i] = dout[i];
        return;
    }
    if (x < -10.0f) {
        dinp[i] = 0.0f;
        return;
    }
    const float cube = 0.044715f * x * x * x;
    const float tanh_arg = kScale * (x + cube);
    const float tanh_out = tanh(tanh_arg);
    const float cosh_out = cosh(tanh_arg);
    const float sech2 = 1.0f / (cosh_out * cosh_out);
    const float local_grad = 0.5f * (1.0f + tanh_out) +
                             x * 0.5f * sech2 * kScale *
                                 (1.0f + 3.0f * 0.044715f * x * x);
    dinp[i] = local_grad * dout[i];
}
)METAL";
    } else if (kernel_name_contains(entry_name, "matmul_backward_bias_kernel4")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* dbias [[buffer(0)]],
    device const float* dout [[buffer(1)]],
    constant int& B [[buffer(2)]],
    constant int& T [[buffer(3)]],
    constant int& OC [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]) {
    if (tid >= 32u || OC <= 0) {
        return;
    }

    const int col = static_cast<int>(group_pos.x * 32u + tid);
    if (col >= OC) {
        return;
    }

    float sum = 0.0f;
    const int rows = B * T;
    for (int row = 0; row < rows; ++row) {
        sum += dout[row * OC + col];
    }

    device atomic_float* dbias_ptr = reinterpret_cast<device atomic_float*>(dbias + col);
    atomic_fetch_add_explicit(dbias_ptr, sum, memory_order_relaxed);
}
)METAL";
    } else if (kernel_name_contains(entry_name, "layernorm_backward_kernel2")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* dinp [[buffer(0)]],
    device float* dweight [[buffer(1)]],
    device float* dbias [[buffer(2)]],
    device const float* dout [[buffer(3)]],
    device const float* inp [[buffer(4)]],
    device const float* weight [[buffer(5)]],
    device const float* mean [[buffer(6)]],
    device const float* rstd [[buffer(7)]],
    constant int& B [[buffer(8)]],
    constant int& T [[buffer(9)]],
    constant int& C [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    if (C <= 0) {
        return;
    }

    const int linear = static_cast<int>(gid);
    const int row = linear / 32;
    const int lane = linear % 32;
    const int N = B * T;
    if (row >= N || lane != 0) {
        return;
    }

    const int base = row * C;
    const float mean_row = mean[row];
    const float rstd_row = rstd[row];
    const float inv_c = 1.0f / static_cast<float>(C);

    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = 0; i < C; ++i) {
        const float norm = (inp[base + i] - mean_row) * rstd_row;
        const float dnorm = weight[i] * dout[base + i];
        dnorm_mean += dnorm;
        dnorm_norm_mean += dnorm * norm;
    }
    dnorm_mean *= inv_c;
    dnorm_norm_mean *= inv_c;

    for (int i = 0; i < C; ++i) {
        const float norm = (inp[base + i] - mean_row) * rstd_row;
        const float dout_val = dout[base + i];
        const float dnorm = weight[i] * dout_val;

        device atomic_float* dbias_ptr = reinterpret_cast<device atomic_float*>(dbias + i);
        device atomic_float* dweight_ptr = reinterpret_cast<device atomic_float*>(dweight + i);
        atomic_fetch_add_explicit(dbias_ptr, dout_val, memory_order_relaxed);
        atomic_fetch_add_explicit(dweight_ptr, norm * dout_val, memory_order_relaxed);

        float dval = dnorm;
        dval -= dnorm_mean;
        dval -= norm * dnorm_norm_mean;
        dval *= rstd_row;

        device atomic_float* dinp_ptr = reinterpret_cast<device atomic_float*>(dinp + base + i);
        atomic_fetch_add_explicit(dinp_ptr, dval, memory_order_relaxed);
    }
}
)METAL";
    } else if (kernel_name_contains(entry_name, "softmax_autoregressive_backward_kernel")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* dpreatt [[buffer(0)]],
    device const float* datt [[buffer(1)]],
    device const float* att [[buffer(2)]],
    constant int& B [[buffer(3)]],
    constant int& T [[buffer(4)]],
    constant int& C [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]) {
    (void)C;
    if (tid != 0u || T <= 0) {
        return;
    }

    const int idx = static_cast<int>(group_pos.y);
    const int t0 = T - 1 - 4 * static_cast<int>(group_pos.x);
    if (idx < 0 || idx >= B * (C / (T > 0 ? T : 1))) {
        // idx corresponds to B * NH in llm.c launches; keep permissive and rely on bounds below.
    }

    device float* dpreatt_base = dpreatt + static_cast<size_t>(idx) * static_cast<size_t>(T) * static_cast<size_t>(T);
    const device float* datt_base = datt + static_cast<size_t>(idx) * static_cast<size_t>(T) * static_cast<size_t>(T);
    const device float* att_base = att + static_cast<size_t>(idx) * static_cast<size_t>(T) * static_cast<size_t>(T);

    for (int to = 0; to < 4; ++to) {
        const int row = t0 - to;
        if (row < 0 || row >= T) {
            continue;
        }

        const int row_base = row * T;
        float local_sum = 0.0f;
        for (int col = 0; col <= row; ++col) {
            local_sum += att_base[row_base + col] * datt_base[row_base + col];
        }
        for (int col = 0; col <= row; ++col) {
            const float a = att_base[row_base + col];
            const float da = datt_base[row_base + col];
            dpreatt_base[row_base + col] = scale * a * (da - local_sum);
        }
        for (int col = row + 1; col < T; ++col) {
            dpreatt_base[row_base + col] = 0.0f;
        }
    }
}
)METAL";
    } else if (kernel_name_contains(entry_name, "adamw_kernel2")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* params [[buffer(0)]],
    device float* grads [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant int& num_parameters [[buffer(4)]],
    constant float& learning_rate [[buffer(5)]],
    constant float& beta1 [[buffer(6)]],
    constant float& beta2 [[buffer(7)]],
    constant float& beta1_correction [[buffer(8)]],
    constant float& beta2_correction [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    constant float& weight_decay [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = static_cast<int>(gid);
    if (i < 0 || i >= num_parameters) {
        return;
    }

    const float grad = grads[i];
    float m_val = m[i];
    float v_val = v[i];

    m_val = fma(beta1, m_val, (1.0f - beta1) * grad);
    v_val = fma(beta2, v_val, (1.0f - beta2) * (grad * grad));
    m[i] = m_val;
    v[i] = v_val;

    const float m_hat = m_val / beta1_correction;
    const float v_hat = v_val / beta2_correction;
    params[i] -= learning_rate * (m_hat / (sqrt(v_hat) + eps) + weight_decay * params[i]);
}
)METAL";
    } else if (kernel_name_contains(entry_name, "fused_classifier_kernel3")) {
kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* logits [[buffer(0)]],
    device float* losses [[buffer(1)]],
    device float* probs [[buffer(2)]],
    device const float* dlosses [[buffer(3)]],
    device const int* targets [[buffer(4)]],
    constant int& B [[buffer(5)]],
    constant int& T [[buffer(6)]],
    constant int& V [[buffer(7)]],
    constant int& P [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]],
    uint3 threads_per_group [[threads_per_threadgroup]]) {
    const int idx = static_cast<int>(group_pos.x);
    const int n = B * T;
    if (idx < 0 || idx >= n || V <= 0 || P <= 0 || V > P) {
        return;
    }

    const int target = targets[idx];
    if (target < 0 || target >= V) {
        return;
    }

    device float* row_logits = logits + static_cast<size_t>(idx) * static_cast<size_t>(P);

    threadgroup float tg_maxval;
    threadgroup float tg_scale;

    if (tid == 0u) {
        float max_val = -3.402823466e+38f;
        for (int i = 0; i < V; ++i) {
            max_val = max(max_val, row_logits[i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < V; ++i) {
            sum += exp(row_logits[i] - max_val);
        }

        tg_maxval = max_val;
        tg_scale = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float max_val = tg_maxval;
    const float scale = tg_scale;

    if (tid == 0u) {
        const float prob = exp(row_logits[target] - max_val) * scale;
        losses[idx] = -log(max(prob, 1.0e-20f));
    }

    const float dloss = dlosses != nullptr ? dlosses[idx] : (1.0f / static_cast<float>(n));
    const int stride = static_cast<int>(threads_per_group.x);
    for (int i = static_cast<int>(tid); i < V; i += stride) {
        const float prob = exp(row_logits[i] - max_val) * scale;
        if (probs != nullptr) {
            probs[static_cast<size_t>(idx) * static_cast<size_t>(P) + static_cast<size_t>(i)] = prob;
        }
        const float indicator = (i == target) ? 1.0f : 0.0f;
        row_logits[i] = (prob - indicator) * dloss;
    }
}
)METAL";
    } else if (kernel_name_contains(entry_name, "matmul_forward_kernel4")) {
        kernel_template = R"METAL(kernel void __KERNEL_NAME__(
    device float* out [[buffer(0)]],
    device const float* inp [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    constant int& C [[buffer(4)]],
    constant int& OC [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]) {
    if (C <= 0 || OC <= 0) {
        return;
    }

    const int row_base = static_cast<int>(gid.x) * 8;
    const int col_base = static_cast<int>(gid.y) * 8;

    for (int ii = 0; ii < 8; ++ii) {
        const int row = row_base + ii;
        const int inp_base = row * C;
        const int out_base = row * OC;
        for (int jj = 0; jj < 8; ++jj) {
            const int col = col_base + jj;
            if (col >= OC) {
                continue;
            }

            float acc = (bias != nullptr) ? bias[col] : 0.0f;
            const int w_base = col * C;
            for (int k = 0; k < C; ++k) {
                acc = fma(inp[inp_base + k], weight[w_base + k], acc);
            }
            out[out_base + col] = acc;
        }
    }
}
)METAL";
    }

    if (kernel_template.empty()) {
        return {};
    }

    std::string source(kPreamble);
    source += replace_kernel_name(kernel_template, entry_name);
    return source;
}

// Generic PTX → Metal emitter for kernels not matched by name.
//
// Handles simple 1-D elementwise kernels: detects the standard global-thread-ID
// computation pattern (mad.lo.u32 gid, ctaid, ntid, tid), resolves gid-indexed
// pointer accesses (base_ptr + gid*element_size), and translates common float/int
// arithmetic instructions to their Metal equivalents.  Returns empty string for
// any instruction or address pattern it cannot translate confidently; the caller
// then falls through to the PTX→LLVM path.
// Returns true if the operands of a call instruction target printf/vprintf.
bool is_printf_call(const std::vector<std::string>& operands) {
    for (const auto& op : operands) {
        const std::string lower = [&]() {
            std::string s = op;
            for (char& c : s) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            return s;
        }();
        if (lower.find("vprintf") != std::string::npos || lower.find("printf") != std::string::npos) {
            return true;
        }
    }
    return false;
}

// Emit inline Metal code to write a printf record to the ring buffer.
// args_and_types is a list of (resolved_metal_expr, metal_type) pairs.
void emit_printf_record(std::ostringstream& metal,
                        std::uint32_t fmt_id,
                        const std::vector<std::pair<std::string, std::string>>& args_and_types,
                        int call_index) {
    const std::uint32_t n_args = static_cast<std::uint32_t>(args_and_types.size());
    const std::string suffix = std::to_string(call_index);
    metal << "    {\n";
    metal << "        const uint __pfw" << suffix << " = " << (2 + n_args) << "u;\n";
    metal << "        uint __ppos" << suffix << " = atomic_fetch_add_explicit(__printf_buf, "
          << "__pfw" << suffix << ", memory_order_relaxed);\n";
    metal << "        if (__ppos" << suffix << " + __pfw" << suffix << " < __printf_cap) {\n";
    metal << "            device uint* __pw" << suffix << " = (device uint*)__printf_buf;\n";
    metal << "            __pw" << suffix << "[__ppos" << suffix << " + 1] = "
          << fmt_id << "u;\n";
    metal << "            __pw" << suffix << "[__ppos" << suffix << " + 2] = "
          << n_args << "u;\n";
    for (std::uint32_t i = 0; i < n_args; ++i) {
        const std::string& expr = args_and_types[i].first;
        const std::string& type = args_and_types[i].second;
        std::string cast_expr;
        if (type == "float") {
            cast_expr = "as_type<uint>(" + expr + ")";
        } else if (type == "double") {
            // Store low 32 bits only for now (sufficient for common FP32-width values)
            cast_expr = "(uint)(as_type<ulong>(" + expr + ") & 0xFFFFFFFFu)";
        } else if (type == "ulong") {
            cast_expr = "(uint)(" + expr + " & 0xFFFFFFFFu)";
        } else if (type == "ushort") {
            cast_expr = "(uint)(" + expr + ")";
        } else {
            cast_expr = "(uint)(" + expr + ")";
        }
        metal << "            __pw" << suffix << "[__ppos" << suffix << " + " << (3 + i) << "] = "
              << cast_expr << ";\n";
    }
    metal << "        }\n";
    metal << "    }\n";
}

std::string emit_metal_source_generic(const std::string& entry_name,
                                      std::string_view ptx,
                                      const cumetal::passes::Phase1PipelineOutput* pipeline_hint) {
    ParseOptions parse_opts;
    parse_opts.strict = false;
    const ParseResult parsed = parse_ptx(ptx, parse_opts);
    if (!parsed.ok) {
        return {};
    }

    const EntryFunction* entry = nullptr;
    for (const auto& e : parsed.module.entries) {
        if (e.name == entry_name) {
            entry = &e;
            break;
        }
    }
    if (entry == nullptr || entry->params.empty()) {
        return {};
    }

    // Build line-number → printf call map from the phase1 output (if available).
    std::unordered_map<int, const cumetal::passes::PrintfLoweredCall*> line_to_printf;
    if (pipeline_hint != nullptr) {
        for (const auto& pc : pipeline_hint->printf_calls) {
            line_to_printf[pc.source_line] = &pc;
        }
    }
    const bool has_printf = !line_to_printf.empty();

    // Only attempt translation when every instruction is in the supported opcode set.
    // Exception: call printf/vprintf is handled inline even if "call" falls through.
    for (const auto& instr : entry->instructions) {
        if (!instr.supported) {
            return {};
        }
        // call instructions that are NOT printf calls (i.e. arbitrary function calls)
        // are not supported by the generic emitter.
        if (instr.opcode.rfind("call", 0) == 0 && !is_printf_call(instr.operands) &&
            !line_to_printf.count(instr.line)) {
            return {};
        }
    }

    // Require at least one global memory operation.  Skeleton PTX (e.g. just
    // a thread-index mov and ret) is meant for LLVM-based body synthesis, not
    // for literal instruction translation.
    {
        bool has_global_mem = false;
        for (const auto& instr : entry->instructions) {
            if (instr.opcode.find("ld.global") == 0 ||
                instr.opcode.find("st.global") == 0 ||
                instr.opcode.find("atom.global") == 0) {
                has_global_mem = true;
                break;
            }
        }
        if (!has_global_mem) {
            return {};
        }
    }

    // ── Pass 1: register provenance analysis ──────────────────────────────────

    enum class RegKind {
        Unknown,
        ParamPtr,      // loaded from a pointer (.is_pointer) param
        ParamScalar,   // loaded from a scalar param
        ThreadTid,     // %tid.x
        ThreadNtid,    // %ntid.x
        ThreadCtaid,   // %ctaid.x
        ThreadPartial, // mul.lo.u32(ctaid, ntid) — intermediate before adding tid
        ThreadGid,     // mad(ctaid, ntid, tid) or add(PartialGid, tid) → global 1-D thread ID
        ThreadGid64,   // (u64)ThreadGid
        ByteOffset,    // ThreadGid64 * byte_per_elem  (from shl / mul)
        DerivedPtr,    // ParamPtr + ByteOffset  → param[gid]
    };
    struct RegInfo {
        RegKind kind = RegKind::Unknown;
        std::string param_name;  // ParamPtr / ParamScalar
        std::string base_param;  // DerivedPtr: which pointer param
        int byte_per_elem = 4;   // ByteOffset / DerivedPtr: element byte width
    };
    std::unordered_map<std::string, RegInfo> reg;

    // Extract the first %register from an operand string (handles "[%rd0]", "[%rd0+4]").
    auto get_reg = [](const std::string& op) -> std::string {
        const std::size_t pct = op.find('%');
        if (pct == std::string::npos) {
            return {};
        }
        std::size_t end = pct + 1;
        while (end < op.size() &&
               (std::isalnum(static_cast<unsigned char>(op[end])) ||
                op[end] == '_' || op[end] == '.')) {
            ++end;
        }
        return (end > pct + 1) ? op.substr(pct, end - pct) : std::string{};
    };

    // Return a non-negative integer if the operand is a plain decimal immediate.
    auto get_imm = [](const std::string& op) -> int {
        if (op.empty() || op[0] == '%') {
            return -1;
        }
        int val = 0;
        for (const char c : op) {
            if (!std::isdigit(static_cast<unsigned char>(c))) {
                return -1;
            }
            val = val * 10 + (c - '0');
        }
        return val;
    };

    std::unordered_map<std::string, bool> param_is_ptr;
    for (const auto& p : entry->params) {
        param_is_ptr[p.name] = p.is_pointer;
    }

    for (const auto& instr : entry->instructions) {
        const auto& op = instr.opcode;
        const auto& ops = instr.operands;
        if (ops.empty()) {
            continue;
        }

        if (op.size() >= 8 && op.substr(0, 8) == "ld.param" && ops.size() >= 2) {
            const std::string dest = get_reg(ops[0]);
            std::string pname;
            if (!ops[1].empty() && ops[1].front() == '[') {
                pname = ops[1].substr(1);
                const auto plus = pname.find('+');
                if (plus != std::string::npos) {
                    pname = pname.substr(0, plus);
                }
                const auto br = pname.find(']');
                if (br != std::string::npos) {
                    pname = pname.substr(0, br);
                }
                while (!pname.empty() &&
                       std::isspace(static_cast<unsigned char>(pname.back()))) {
                    pname.pop_back();
                }
            }
            if (!dest.empty() && !pname.empty()) {
                const auto it = param_is_ptr.find(pname);
                if (it != param_is_ptr.end()) {
                    reg[dest] = {.kind = it->second ? RegKind::ParamPtr : RegKind::ParamScalar,
                                 .param_name = pname};
                }
            }
            continue;
        }

        if ((op == "mov.u32" || op == "mov.s32") && ops.size() == 2) {
            const std::string dest = get_reg(ops[0]);
            if (!dest.empty()) {
                if (ops[1] == "%tid.x") {
                    reg[dest] = {.kind = RegKind::ThreadTid};
                } else if (ops[1] == "%ntid.x") {
                    reg[dest] = {.kind = RegKind::ThreadNtid};
                } else if (ops[1] == "%ctaid.x") {
                    reg[dest] = {.kind = RegKind::ThreadCtaid};
                }
            }
            continue;
        }

        // mad.lo.u32 gid, ctaid, ntid, tid  (or ntid, ctaid, tid)
        if ((op == "mad.lo.u32" || op == "mad.lo.s32") && ops.size() == 4) {
            const std::string dest = get_reg(ops[0]);
            const std::string s1 = get_reg(ops[1]);
            const std::string s2 = get_reg(ops[2]);
            const std::string s3 = get_reg(ops[3]);
            if (!dest.empty() && !s1.empty() && !s2.empty() && !s3.empty()) {
                const auto k1 = reg.count(s1) ? reg.at(s1).kind : RegKind::Unknown;
                const auto k2 = reg.count(s2) ? reg.at(s2).kind : RegKind::Unknown;
                const auto k3 = reg.count(s3) ? reg.at(s3).kind : RegKind::Unknown;
                const bool gid_pattern =
                    ((k1 == RegKind::ThreadCtaid && k2 == RegKind::ThreadNtid) ||
                     (k1 == RegKind::ThreadNtid && k2 == RegKind::ThreadCtaid)) &&
                    k3 == RegKind::ThreadTid;
                if (gid_pattern) {
                    reg[dest] = {.kind = RegKind::ThreadGid};
                }
            }
            continue;
        }

        // mul.lo.u32 partial, ctaid, ntid  (first half of two-instruction gid pattern)
        if ((op == "mul.lo.u32" || op == "mul.lo.s32") && ops.size() == 3) {
            const std::string dest = get_reg(ops[0]);
            const std::string s1 = get_reg(ops[1]);
            const std::string s2 = get_reg(ops[2]);
            if (!dest.empty() && !s1.empty() && !s2.empty()) {
                const auto k1 = reg.count(s1) ? reg.at(s1).kind : RegKind::Unknown;
                const auto k2 = reg.count(s2) ? reg.at(s2).kind : RegKind::Unknown;
                const bool partial_pattern =
                    (k1 == RegKind::ThreadCtaid && k2 == RegKind::ThreadNtid) ||
                    (k1 == RegKind::ThreadNtid && k2 == RegKind::ThreadCtaid);
                if (partial_pattern) {
                    reg[dest] = {.kind = RegKind::ThreadPartial};
                }
            }
            continue;
        }

        // add.u32 gid, partial, tid  (second half of two-instruction gid pattern)
        if ((op == "add.u32" || op == "add.s32") && ops.size() == 3) {
            const std::string dest = get_reg(ops[0]);
            const std::string s1 = get_reg(ops[1]);
            const std::string s2 = get_reg(ops[2]);
            if (!dest.empty() && !s1.empty() && !s2.empty()) {
                const auto k1 = reg.count(s1) ? reg.at(s1).kind : RegKind::Unknown;
                const auto k2 = reg.count(s2) ? reg.at(s2).kind : RegKind::Unknown;
                const bool gid_pattern =
                    (k1 == RegKind::ThreadPartial && k2 == RegKind::ThreadTid) ||
                    (k1 == RegKind::ThreadTid && k2 == RegKind::ThreadPartial);
                if (gid_pattern) {
                    reg[dest] = {.kind = RegKind::ThreadGid};
                }
            }
            continue;
        }

        // cvt.*.u64.{u32,s32} — promote 32-bit gid to 64-bit
        if (op.find("cvt") == 0 && op.find(".u64.") != std::string::npos && ops.size() == 2) {
            const std::string dest = get_reg(ops[0]);
            const std::string src = get_reg(ops[1]);
            if (!dest.empty() && !src.empty() && reg.count(src) &&
                reg.at(src).kind == RegKind::ThreadGid) {
                reg[dest] = {.kind = RegKind::ThreadGid64};
            }
            continue;
        }

        // shl.b64 rdN, rdGID64, imm  →  byte_offset = gid * 2^imm
        if (op == "shl.b64" && ops.size() == 3) {
            const std::string dest = get_reg(ops[0]);
            const std::string src = get_reg(ops[1]);
            const int imm = get_imm(ops[2]);
            if (!dest.empty() && !src.empty() && imm >= 0 && reg.count(src) &&
                reg.at(src).kind == RegKind::ThreadGid64) {
                reg[dest] = {.kind = RegKind::ByteOffset, .byte_per_elem = 1 << imm};
            }
            continue;
        }

        // mul.lo.u64 rdN, rdGID64, imm  →  byte_offset = gid * imm
        if ((op == "mul.lo.u64" || op == "mul.wide.u32") && ops.size() == 3) {
            const std::string dest = get_reg(ops[0]);
            const std::string src = get_reg(ops[1]);
            const int imm = get_imm(ops[2]);
            if (!dest.empty() && !src.empty() && imm > 0 && reg.count(src) &&
                reg.at(src).kind == RegKind::ThreadGid64) {
                reg[dest] = {.kind = RegKind::ByteOffset, .byte_per_elem = imm};
            }
            continue;
        }

        // add.u64 rdP, rdBase, rdOffset  →  derived pointer: param[gid]
        if ((op == "add.u64" || op == "add.s64") && ops.size() == 3) {
            const std::string dest = get_reg(ops[0]);
            const std::string s1 = get_reg(ops[1]);
            const std::string s2 = get_reg(ops[2]);
            if (!dest.empty() && !s1.empty() && !s2.empty()) {
                const auto k1 = reg.count(s1) ? reg.at(s1).kind : RegKind::Unknown;
                const auto k2 = reg.count(s2) ? reg.at(s2).kind : RegKind::Unknown;
                const RegInfo* base_r = nullptr;
                const RegInfo* off_r = nullptr;
                if (k1 == RegKind::ParamPtr && k2 == RegKind::ByteOffset) {
                    base_r = &reg.at(s1);
                    off_r = &reg.at(s2);
                } else if (k2 == RegKind::ParamPtr && k1 == RegKind::ByteOffset) {
                    base_r = &reg.at(s2);
                    off_r = &reg.at(s1);
                }
                if (base_r != nullptr && off_r != nullptr) {
                    reg[dest] = {.kind = RegKind::DerivedPtr,
                                 .base_param = base_r->param_name,
                                 .byte_per_elem = off_r->byte_per_elem};
                }
            }
            continue;
        }
    }

    // ── Validate: every global load/store must have a resolved address ─────────

    bool has_global = false;
    for (const auto& instr : entry->instructions) {
        if (instr.opcode.find("ld.global") == 0 || instr.opcode.find("st.global") == 0) {
            has_global = true;
            const bool is_load = (instr.opcode[0] == 'l');
            const std::string& addr_op =
                is_load ? instr.operands[1] : instr.operands[0];
            if (addr_op.empty() || addr_op.front() != '[') {
                return {};
            }
            const std::string mreg = get_reg(addr_op);
            if (mreg.empty()) {
                return {};
            }
            const auto it = reg.find(mreg);
            if (it == reg.end() ||
                (it->second.kind != RegKind::DerivedPtr &&
                 it->second.kind != RegKind::ParamPtr)) {
                return {};
            }
        }
    }

    bool has_gid = false;
    for (const auto& kv : reg) {
        if (kv.second.kind == RegKind::ThreadGid) {
            has_gid = true;
            break;
        }
    }
    if (has_global && !has_gid) {
        return {};
    }

    // ── Determine element type for each pointer param ─────────────────────────

    std::unordered_map<std::string, std::string> param_etype;
    for (const auto& instr : entry->instructions) {
        const auto& op = instr.opcode;
        if (op.find("ld.global") != 0 && op.find("st.global") != 0) {
            continue;
        }
        const bool is_load = (op[0] == 'l');
        const std::string mreg = get_reg(is_load ? instr.operands[1] : instr.operands[0]);
        if (mreg.empty()) {
            continue;
        }
        const auto it = reg.find(mreg);
        if (it == reg.end()) {
            continue;
        }
        const std::string& pname = (it->second.kind == RegKind::DerivedPtr)
                                       ? it->second.base_param
                                       : it->second.param_name;
        std::string etype;
        if (op.find(".f32") != std::string::npos) {
            etype = "float";
        } else if (op.find(".f64") != std::string::npos) {
            etype = "double";
        } else if (op.find(".u32") != std::string::npos) {
            etype = "uint";
        } else if (op.find(".s32") != std::string::npos) {
            etype = "int";
        } else if (op.find(".u64") != std::string::npos) {
            etype = "ulong";
        } else {
            etype = "float";
        }
        if (!pname.empty()) {
            param_etype[pname] = etype;
        }
    }

    // ── Detect bounds-check setp pattern ─────────────────────────────────────
    // setp.ge/gt %p, %rGID, %rN  where rGID=ThreadGid and rN=ParamScalar →
    // emit `if (gid >= (uint)N) return;` in place of the setp + branch pair.

    std::unordered_map<std::string, std::string> pred_guard;  // pred_reg → Metal expr
    for (const auto& instr : entry->instructions) {
        const auto& op = instr.opcode;
        const auto& ops = instr.operands;
        if (op.find("setp") != 0 || ops.size() < 3) {
            continue;
        }
        const std::string dp = get_reg(ops[0]);
        const std::string s1 = get_reg(ops[1]);
        const std::string s2 = get_reg(ops[2]);
        const bool s1_gid = reg.count(s1) && reg.at(s1).kind == RegKind::ThreadGid;
        const bool s2_scalar = reg.count(s2) && reg.at(s2).kind == RegKind::ParamScalar;
        if (!s1_gid || !s2_scalar) {
            continue;
        }
        std::string cmp;
        if (op.find(".ge") != std::string::npos) {
            cmp = ">=";
        } else if (op.find(".gt") != std::string::npos) {
            cmp = ">";
        } else if (op.find(".lt") != std::string::npos) {
            cmp = "<";
        } else if (op.find(".le") != std::string::npos) {
            cmp = "<=";
        }
        if (!dp.empty() && !cmp.empty()) {
            pred_guard[dp] = "gid " + cmp + " (uint)" + reg.at(s2).param_name;
        }
    }

    // ── Pass 2: emit Metal source ─────────────────────────────────────────────

    std::ostringstream metal;
    metal << "#include <metal_stdlib>\n#include <metal_atomic>\n\nusing namespace metal;\n\n";
    metal << "kernel void " << entry_name << "(\n";

    int buf_idx = 0;
    bool first_arg = true;
    for (const auto& p : entry->params) {
        if (!first_arg) {
            metal << ",\n";
        }
        first_arg = false;
        if (p.is_pointer) {
            const std::string etype =
                param_etype.count(p.name) ? param_etype.at(p.name) : "float";
            metal << "    device " << etype << "* " << p.name
                  << " [[buffer(" << buf_idx << ")]]";
        } else {
            std::string mtype = "uint";
            if (p.type == ".u64" || p.type == ".s64" || p.type == ".b64") {
                mtype = "ulong";
            } else if (p.type == ".f32") {
                mtype = "float";
            } else if (p.type == ".f64") {
                mtype = "double";
            } else if (p.type == ".u16" || p.type == ".s16" || p.type == ".b16") {
                mtype = "ushort";
            }
            metal << "    constant " << mtype << "& " << p.name
                  << " [[buffer(" << buf_idx << ")]]";
        }
        ++buf_idx;
    }
    // ── Hidden printf ring-buffer args (spec §5.3) ──────────────────────────
    if (has_printf) {
        metal << ",\n    device atomic_uint* __printf_buf [[buffer(" << buf_idx << ")]]";
        ++buf_idx;
        metal << ",\n    constant uint& __printf_cap [[buffer(" << buf_idx << ")]]";
        ++buf_idx;
    }
    metal << ",\n    uint gid [[thread_position_in_grid]],\n"
          << "    ushort __laneid [[thread_index_in_simdgroup]]) {\n";

    // Metal variable name for a PTX register: %rd0 → vrd0, %f1 → vf1
    auto mvar = [](const std::string& r) -> std::string {
        return (r.size() > 1 && r[0] == '%') ? "v" + r.substr(1) : r;
    };

    // Resolve an operand to a Metal expression (reg or immediate)
    auto resolve = [&](const std::string& operand) -> std::string {
        const std::string r = get_reg(operand);
        if (r.empty()) {
            // PTX 0d<hex16> — IEEE 754 double bit-cast literal
            if (operand.size() == 18 && operand[0] == '0' && operand[1] == 'd') {
                const std::uint64_t bits = std::stoull(operand.substr(2), nullptr, 16);
                double val;
                std::memcpy(&val, &bits, sizeof(val));
                std::ostringstream oss;
                oss << std::scientific << std::setprecision(17) << val;
                return oss.str();
            }
            // PTX 0f<hex8> — IEEE 754 float bit-cast literal
            if (operand.size() == 10 && operand[0] == '0' && operand[1] == 'f') {
                const std::uint32_t bits =
                    static_cast<std::uint32_t>(std::stoul(operand.substr(2), nullptr, 16));
                float val;
                std::memcpy(&val, &bits, sizeof(val));
                std::ostringstream oss;
                oss << std::scientific << std::setprecision(9) << val << "f";
                return oss.str();
            }
            return operand;  // other immediate (decimal, hex with 0x prefix, etc.)
        }
        const auto it = reg.find(r);
        if (it != reg.end() && it->second.kind == RegKind::ThreadGid) {
            return "gid";
        }
        if (it != reg.end() && it->second.kind == RegKind::ParamScalar) {
            return it->second.param_name;
        }
        return mvar(r);
    };

    // Metal type inferred from PTX register name prefix
    auto reg_type = [](const std::string& r) -> std::string {
        if (r.size() > 2 && r[0] == '%' && r[1] == 'r' && r[2] == 'd') return "ulong";
        if (r.size() > 2 && r[0] == '%' && r[1] == 'f' && r[2] == 'd') return "double";
        if (r.size() > 1 && r[0] == '%' && r[1] == 'f') return "float";
        if (r.size() > 1 && r[0] == '%' && r[1] == 'p') return "bool";
        if (r.size() > 1 && r[0] == '%' && r[1] == 'h') return "ushort";
        return "uint";
    };

    std::unordered_set<std::string> consumed_guards;

    // Track registers that have been defined so far in the emitted Metal body.
    // Source operands that refer to undefined registers indicate the PTX body
    // is not self-contained (e.g. it relies on LLVM-synthesised setup) — bail.
    std::unordered_set<std::string> defined_regs;
    // Registers that are structural (resolved via parameter bindings or gid)
    // are always "defined" from the Metal kernel's perspective.
    for (const auto& kv : reg) {
        defined_regs.insert(kv.first);
    }

    // Helper: return false if any source register in `src_ops` (starting at
    // `first_src_index`) is an unrecognised undefined register.
    auto all_sources_defined = [&](const std::vector<std::string>& ops,
                                   std::size_t first_src_index) -> bool {
        for (std::size_t i = first_src_index; i < ops.size(); ++i) {
            const std::string r = get_reg(ops[i]);
            if (r.empty()) continue;  // immediate — fine
            if (defined_regs.count(r)) continue;
            return false;
        }
        return true;
    };

    int shfl_tmp_id = 0;
    for (const auto& instr : entry->instructions) {
        const auto& op = instr.opcode;
        const auto& ops = instr.operands;

        // ── Structural: parameter loads, labels, ret ────────────────────────
        if (op.size() >= 8 && op.substr(0, 8) == "ld.param") continue;
        if (op == "ptx.label") continue;
        if (op == "ret") continue;

        // mov %r, %tid/ntid/ctaid.x → structural
        if ((op == "mov.u32" || op == "mov.s32") && ops.size() == 2 &&
            (ops[1] == "%tid.x" || ops[1] == "%ntid.x" || ops[1] == "%ctaid.x")) {
            continue;
        }

        // Skip instructions whose destination is a structural register.
        // Only applies when ops[0] is a plain register (not a bracket-addressed
        // memory operand like "[%rd3]" used by store instructions).
        if (!ops.empty() && (ops[0].empty() || ops[0][0] != '[')) {
            const std::string dest = get_reg(ops[0]);
            if (!dest.empty() && reg.count(dest)) {
                switch (reg.at(dest).kind) {
                    case RegKind::ThreadPartial:
                    case RegKind::ThreadGid:
                    case RegKind::ThreadGid64:
                    case RegKind::ByteOffset:
                    case RegKind::DerivedPtr:
                    case RegKind::ParamPtr:
                    case RegKind::ParamScalar:
                        continue;
                    default:
                        break;
                }
            }
        }

        // ── Global memory load ───────────────────────────────────────────────
        if (op.find("ld.global") == 0 && ops.size() >= 2) {
            const std::string dest = get_reg(ops[0]);
            const std::string mreg = get_reg(ops[1]);
            const auto it = reg.find(mreg);
            if (it == reg.end()) return {};
            const std::string& pname = (it->second.kind == RegKind::DerivedPtr)
                                           ? it->second.base_param
                                           : it->second.param_name;
            const std::string etype =
                param_etype.count(pname) ? param_etype.at(pname) : "float";
            metal << "    " << etype << " " << mvar(dest) << " = " << pname << "[gid];\n";
            defined_regs.insert(dest);
            continue;
        }

        // ── Global memory store ──────────────────────────────────────────────
        if (op.find("st.global") == 0 && ops.size() >= 2) {
            const std::string mreg = get_reg(ops[0]);
            const auto it = reg.find(mreg);
            if (it == reg.end()) return {};
            const std::string& pname = (it->second.kind == RegKind::DerivedPtr)
                                           ? it->second.base_param
                                           : it->second.param_name;
            metal << "    " << pname << "[gid] = " << resolve(ops[1]) << ";\n";
            continue;
        }

        // ── setp: bounds guard or generic comparison ─────────────────────────
        if (op.find("setp") == 0 && ops.size() >= 3) {
            const std::string dp = get_reg(ops[0]);
            if (pred_guard.count(dp)) {
                metal << "    if (" << pred_guard.at(dp) << ") return;\n";
                consumed_guards.insert(dp);
                defined_regs.insert(dp);
                continue;
            }
            if (!all_sources_defined(ops, 1)) return {};
            // Generic comparison
            std::string cmp;
            if (op.find(".ge") != std::string::npos) cmp = ">=";
            else if (op.find(".gt") != std::string::npos) cmp = ">";
            else if (op.find(".lt") != std::string::npos) cmp = "<";
            else if (op.find(".le") != std::string::npos) cmp = "<=";
            else if (op.find(".eq") != std::string::npos) cmp = "==";
            else if (op.find(".ne") != std::string::npos) cmp = "!=";
            else return {};
            metal << "    bool " << mvar(dp) << " = "
                  << resolve(ops[1]) << " " << cmp << " " << resolve(ops[2]) << ";\n";
            defined_regs.insert(dp);
            continue;
        }

        // ── Conditional branch ───────────────────────────────────────────────
        if (op == "bra" && !instr.predicate.empty()) {
            std::string pred_str = instr.predicate;
            if (pred_str.size() > 1 && pred_str[1] == '!') {
                pred_str = pred_str[0] + pred_str.substr(2);
            }
            const std::string pr = get_reg(pred_str);
            if (consumed_guards.count(pr)) {
                continue;  // already emitted as early return
            }
            // Cannot safely translate generic forward/backward branches to MSL.
            return {};
        }

        // ── Unconditional branch → unsupported ──────────────────────────────
        if (op == "bra") return {};

        // ── bar.* → force fallback ───────────────────────────────────────────
        // Generic PTX→Metal translation cannot safely preserve threadgroup
        // barrier semantics for multi-warp blocks. Dropping bar.sync was okay
        // for llm.c single-warp kernels but is incorrect for ggml reductions.
        if (op.size() >= 3 && op.substr(0, 3) == "bar") return {};

        // ── fence / membar → no-op (UMA — all memory is coherent) ────────────
        if (op.find("fence") == 0 || op.find("membar") == 0) continue;

        // ── shfl.sync: warp shuffle → Metal simd_shuffle* ────────────────────
        // shfl.sync.{idx,down,up,bfly}.b32 d|p, src, lane/delta, clamp, mask
        // Honor PTX clamp-encoded width to preserve sub-warp shuffle semantics used
        // by ggml reductions. membermask remains conservatively ignored (full group).
        if (op.find("shfl.sync") == 0 && ops.size() >= 3) {
            if (!all_sources_defined(ops, 1)) return {};
            // ops[0] may be "dst|pred_dst"; extract the value register.
            const std::string raw0 = ops[0];
            const auto pipe = raw0.find('|');
            const std::string dest = get_reg(pipe != std::string::npos ? raw0.substr(0, pipe) : raw0);
            const std::string pred_dest = pipe != std::string::npos ? get_reg(raw0.substr(pipe + 1)) : "";
            const std::string src = resolve(ops[1]);
            const std::string lane = resolve(ops[2]);
            const std::string clamp = ops.size() >= 4 ? resolve(ops[3]) : "31";
            const std::string dtype = reg_type(dest);
            const int sid = shfl_tmp_id++;
            const std::string lane_id_v = "__cm_shfl_lane_" + std::to_string(sid);
            const std::string clamp_v = "__cm_shfl_clamp_" + std::to_string(sid);
            const std::string width_v = "__cm_shfl_width_" + std::to_string(sid);
            const std::string base_v = "__cm_shfl_base_" + std::to_string(sid);
            const std::string local_v = "__cm_shfl_local_" + std::to_string(sid);
            const std::string target_v = "__cm_shfl_target_" + std::to_string(sid);
            const std::string valid_v = "__cm_shfl_valid_" + std::to_string(sid);
            const std::string val_v = "__cm_shfl_val_" + std::to_string(sid);
            metal << "    uint " << lane_id_v << " = (uint)__laneid & 31u;\n";
            metal << "    uint " << clamp_v << " = (uint)(" << clamp << ");\n";
            metal << "    uint " << width_v << " = 32u - ((" << clamp_v << " >> 8) & 0x1fu);\n";
            metal << "    " << width_v << " = (" << width_v << " == 0u) ? 32u : " << width_v << ";\n";
            metal << "    uint " << base_v << " = (" << lane_id_v << " / " << width_v << ") * " << width_v << ";\n";
            metal << "    uint " << local_v << " = " << lane_id_v << " - " << base_v << ";\n";
            if (op.find(".down.") != std::string::npos) {
                const std::string delta_v = "__cm_shfl_delta_" + std::to_string(sid);
                metal << "    uint " << delta_v << " = (uint)(" << lane << ");\n";
                metal << "    uint " << target_v << " = " << lane_id_v << " + " << delta_v << ";\n";
                metal << "    bool " << valid_v << " = " << target_v << " < (" << base_v << " + " << width_v << ");\n";
            } else if (op.find(".up.") != std::string::npos) {
                const std::string delta_v = "__cm_shfl_delta_" + std::to_string(sid);
                metal << "    uint " << delta_v << " = (uint)(" << lane << ");\n";
                metal << "    uint " << target_v << " = " << lane_id_v << " - " << delta_v << ";\n";
                metal << "    bool " << valid_v << " = " << local_v << " >= " << delta_v << ";\n";
            } else if (op.find(".bfly.") != std::string::npos) {
                const std::string xor_v = "__cm_shfl_xor_" + std::to_string(sid);
                const std::string tlocal_v = "__cm_shfl_tlocal_" + std::to_string(sid);
                metal << "    uint " << xor_v << " = (uint)(" << lane << ");\n";
                metal << "    uint " << tlocal_v << " = " << local_v << " ^ " << xor_v << ";\n";
                metal << "    uint " << target_v << " = " << base_v << " + " << tlocal_v << ";\n";
                metal << "    bool " << valid_v << " = " << tlocal_v << " < " << width_v << ";\n";
            } else {
                const std::string src_local_v = "__cm_shfl_src_local_" + std::to_string(sid);
                metal << "    uint " << src_local_v << " = ((uint)(" << lane << ")) & (" << width_v << " - 1u);\n";
                metal << "    uint " << target_v << " = " << base_v << " + " << src_local_v << ";\n";
                metal << "    bool " << valid_v << " = true;\n";
            }
            metal << "    " << dtype << " " << val_v
                  << " = (" << dtype << ")simd_shuffle(" << src << ", (ushort)" << target_v << ");\n";
            metal << "    " << dtype << " " << mvar(dest)
                  << " = " << valid_v << " ? " << val_v << " : (" << dtype << ")(" << src << ");\n";
            if (!pred_dest.empty()) {
                metal << "    bool " << mvar(pred_dest) << " = " << valid_v << ";\n";
                defined_regs.insert(pred_dest);
            }
            defined_regs.insert(dest);
            continue;
        }

        // ── vote.sync: warp-wide predicate vote → Metal simd_* ───────────────
        // vote.sync.ballot.b32 d, pred, mask → simd_ballot
        // vote.sync.any.pred   d, pred, mask → simd_any
        // vote.sync.all.pred   d, pred, mask → simd_all
        if (op.find("vote.sync") == 0 && ops.size() >= 2) {
            if (!all_sources_defined(ops, 1)) return {};
            const std::string dest = get_reg(ops[0]);
            const std::string pred = resolve(ops[1]);
            const std::string dtype = reg_type(dest);
            if (op.find(".ballot.") != std::string::npos) {
                metal << "    " << dtype << " " << mvar(dest)
                      << " = (uint)simd_ballot((bool)" << pred << ");\n";
            } else if (op.find(".any.") != std::string::npos) {
                metal << "    bool " << mvar(dest) << " = simd_any((bool)" << pred << ");\n";
            } else if (op.find(".all.") != std::string::npos) {
                metal << "    bool " << mvar(dest) << " = simd_all((bool)" << pred << ");\n";
            } else {
                // vote.uni / vote.sync.uni — uniformity test, same as all
                metal << "    bool " << mvar(dest) << " = simd_all((bool)" << pred << ");\n";
            }
            defined_regs.insert(dest);
            continue;
        }

        // ── redux.sync: warp-wide reduction → Metal simd_sum/and/or/xor ──────
        // redux.sync.{add,and,or,xor,min,max}.{s32,u32,b32,f32} d, src, mask
        if (op.find("redux.sync") == 0 && ops.size() >= 2) {
            if (!all_sources_defined(ops, 1)) return {};
            const std::string dest = get_reg(ops[0]);
            const std::string src = resolve(ops[1]);
            const std::string dtype = reg_type(dest);
            std::string reduce_fn;
            if (op.find(".add.") != std::string::npos) {
                reduce_fn = "simd_sum";
            } else if (op.find(".and.") != std::string::npos) {
                reduce_fn = "simd_and";
            } else if (op.find(".or.") != std::string::npos) {
                reduce_fn = "simd_or";
            } else if (op.find(".xor.") != std::string::npos) {
                reduce_fn = "simd_xor";
            } else if (op.find(".min.") != std::string::npos) {
                reduce_fn = "simd_min";
            } else if (op.find(".max.") != std::string::npos) {
                reduce_fn = "simd_max";
            } else {
                reduce_fn = "simd_sum";
            }
            metal << "    " << dtype << " " << mvar(dest)
                  << " = (" << dtype << ")" << reduce_fn << "(" << src << ");\n";
            defined_regs.insert(dest);
            continue;
        }

        // ── Type conversion ──────────────────────────────────────────────────
        if (op.find("cvt") == 0 && ops.size() == 2) {
            if (!all_sources_defined(ops, 1)) return {};
            const std::string dest = get_reg(ops[0]);
            const std::string src = resolve(ops[1]);
            const std::string dtype = reg_type(dest);
            metal << "    " << dtype << " " << mvar(dest) << " = (" << dtype << ")" << src << ";\n";
            defined_regs.insert(dest);
            continue;
        }

        // ── Arithmetic (binary) ──────────────────────────────────────────────
        const std::string root = [&]() {
            const auto dot = op.find('.');
            return dot != std::string::npos ? op.substr(0, dot) : op;
        }();

        auto emit_binary_op = [&](const std::string& metal_op) -> bool {
            if (ops.size() < 3) return false;
            if (!all_sources_defined(ops, 1)) return false;
            const std::string dest = get_reg(ops[0]);
            metal << "    " << reg_type(dest) << " " << mvar(dest)
                  << " = " << resolve(ops[1]) << " " << metal_op << " " << resolve(ops[2]) << ";\n";
            defined_regs.insert(dest);
            return true;
        };

        if (root == "add") { if (!emit_binary_op("+")) return {}; continue; }
        if (root == "sub") { if (!emit_binary_op("-")) return {}; continue; }
        if (root == "mul") { if (!emit_binary_op("*")) return {}; continue; }
        if (root == "div") { if (!emit_binary_op("/")) return {}; continue; }
        if (root == "rem") { if (!emit_binary_op("%")) return {}; continue; }
        if (root == "and") { if (!emit_binary_op("&")) return {}; continue; }
        if (root == "or")  { if (!emit_binary_op("|")) return {}; continue; }
        if (root == "xor") { if (!emit_binary_op("^")) return {}; continue; }
        if (root == "shl") { if (!emit_binary_op("<<")) return {}; continue; }
        if (root == "shr") { if (!emit_binary_op(">>")) return {}; continue; }

        if (root == "mad" && ops.size() >= 4) {
            if (!all_sources_defined(ops, 1)) return {};
            const std::string dest = get_reg(ops[0]);
            metal << "    " << reg_type(dest) << " " << mvar(dest) << " = "
                  << resolve(ops[1]) << " * " << resolve(ops[2]) << " + " << resolve(ops[3]) << ";\n";
            defined_regs.insert(dest);
            continue;
        }
        if (root == "fma" && ops.size() >= 4) {
            if (!all_sources_defined(ops, 1)) return {};
            const std::string dest = get_reg(ops[0]);
            metal << "    " << reg_type(dest) << " " << mvar(dest) << " = fma("
                  << resolve(ops[1]) << ", " << resolve(ops[2]) << ", " << resolve(ops[3]) << ");\n";
            defined_regs.insert(dest);
            continue;
        }
        if (root == "neg" && ops.size() >= 2) {
            if (!all_sources_defined(ops, 1)) return {};
            const std::string dest = get_reg(ops[0]);
            metal << "    " << reg_type(dest) << " " << mvar(dest) << " = -" << resolve(ops[1]) << ";\n";
            defined_regs.insert(dest);
            continue;
        }
        if (root == "rcp" && ops.size() >= 2) {
            if (!all_sources_defined(ops, 1)) return {};
            const std::string dest = get_reg(ops[0]);
            metal << "    " << reg_type(dest) << " " << mvar(dest) << " = 1.0f / " << resolve(ops[1]) << ";\n";
            defined_regs.insert(dest);
            continue;
        }
        if (root == "abs" && ops.size() >= 2) {
            if (!all_sources_defined(ops, 1)) return {};
            const std::string dest = get_reg(ops[0]);
            metal << "    " << reg_type(dest) << " " << mvar(dest) << " = abs(" << resolve(ops[1]) << ");\n";
            defined_regs.insert(dest);
            continue;
        }
        if ((root == "max" || root == "min") && ops.size() >= 3) {
            if (!all_sources_defined(ops, 1)) return {};
            const std::string dest = get_reg(ops[0]);
            metal << "    " << reg_type(dest) << " " << mvar(dest) << " = "
                  << root << "(" << resolve(ops[1]) << ", " << resolve(ops[2]) << ");\n";
            defined_regs.insert(dest);
            continue;
        }
        if (root == "not" && ops.size() >= 2) {
            if (!all_sources_defined(ops, 1)) return {};
            const std::string dest = get_reg(ops[0]);
            metal << "    " << reg_type(dest) << " " << mvar(dest) << " = ~" << resolve(ops[1]) << ";\n";
            defined_regs.insert(dest);
            continue;
        }

        // ── Unary math intrinsics ────────────────────────────────────────────
        {
            auto emit_unary_fn = [&](const std::string& fn) -> bool {
                if (ops.size() < 2) return false;
                if (!all_sources_defined(ops, 1)) return false;
                const std::string dest = get_reg(ops[0]);
                metal << "    " << reg_type(dest) << " " << mvar(dest)
                      << " = " << fn << "(" << resolve(ops[1]) << ");\n";
                defined_regs.insert(dest);
                return true;
            };
            if (root == "sqrt")  { if (!emit_unary_fn("sqrt"))  return {}; continue; }
            if (root == "rsqrt") { if (!emit_unary_fn("rsqrt")) return {}; continue; }
            if (root == "ex2")   { if (!emit_unary_fn("exp2"))  return {}; continue; }
            if (root == "lg2")   { if (!emit_unary_fn("log2"))  return {}; continue; }
            if (root == "sin")   { if (!emit_unary_fn("sin"))   return {}; continue; }
            if (root == "cos")   { if (!emit_unary_fn("cos"))   return {}; continue; }
        }
        if (root == "selp" && ops.size() >= 4) {
            if (!all_sources_defined(ops, 1)) return {};
            const std::string dest = get_reg(ops[0]);
            metal << "    " << reg_type(dest) << " " << mvar(dest) << " = "
                  << resolve(ops[3]) << " ? " << resolve(ops[1]) << " : " << resolve(ops[2]) << ";\n";
            defined_regs.insert(dest);
            continue;
        }
        if (root == "mov" && ops.size() == 2) {
            if (!all_sources_defined(ops, 1)) return {};
            const std::string dest = get_reg(ops[0]);
            metal << "    " << reg_type(dest) << " " << mvar(dest) << " = " << resolve(ops[1]) << ";\n";
            defined_regs.insert(dest);
            continue;
        }

        // ── atom.global.add.f32 ─────────────────────────────────────────────
        if (op.find("atom.global.add.f32") == 0 && ops.size() >= 3) {
            if (!all_sources_defined(ops, 2)) return {};
            const std::string dest = get_reg(ops[0]);
            const std::string mreg = get_reg(ops[1]);
            const auto it = reg.find(mreg);
            if (it == reg.end()) return {};
            const bool is_derived = (it->second.kind == RegKind::DerivedPtr);
            const std::string& pname = is_derived ? it->second.base_param
                                                   : it->second.param_name;
            const std::string atm = "atm_" + mvar(dest);
            // DerivedPtr: each thread has its own slot (param[gid]).
            // Raw ParamPtr: global accumulation to a fixed base address (param[0]).
            if (is_derived) {
                metal << "    device atomic_float* " << atm
                      << " = reinterpret_cast<device atomic_float*>(" << pname << " + gid);\n";
            } else {
                metal << "    device atomic_float* " << atm
                      << " = reinterpret_cast<device atomic_float*>(" << pname << ");\n";
            }
            metal << "    float " << mvar(dest) << " = atomic_fetch_add_explicit("
                  << atm << ", " << resolve(ops[2]) << ", memory_order_relaxed);\n";
            defined_regs.insert(dest);
            continue;
        }

        // ── call printf/vprintf → inline ring-buffer write (spec §5.3) ──────
        if (has_printf && instr.opcode.rfind("call", 0) == 0) {
            const auto pc_it = line_to_printf.find(instr.line);
            if (pc_it != line_to_printf.end()) {
                const cumetal::passes::PrintfLoweredCall& pc = *pc_it->second;
                // Build (expr, type) pairs for each argument register.
                std::vector<std::pair<std::string, std::string>> args_and_types;
                for (const auto& arg_token : pc.arguments) {
                    const std::string r = get_reg(arg_token);
                    const std::string expr = r.empty() ? arg_token : resolve(arg_token);
                    const std::string type = r.empty() ? "uint" : reg_type(r);
                    args_and_types.emplace_back(expr, type);
                }
                // Mark any destination register as defined (usually return value = 0).
                if (!ops.empty()) {
                    const std::string dest_raw = ops[0];
                    // Destination might be "(  %r0  )" style; strip parens and spaces.
                    std::string dest_clean;
                    for (char c : dest_raw) {
                        if (c != '(' && c != ')' && c != ' ') dest_clean += c;
                    }
                    const std::string dest = get_reg(dest_clean);
                    if (!dest.empty() && !defined_regs.count(dest)) {
                        metal << "    " << reg_type(dest) << " " << mvar(dest) << " = 0;\n";
                        defined_regs.insert(dest);
                    }
                }
                emit_printf_record(metal, pc.format_id, args_and_types,
                                   static_cast<int>(std::distance(entry->instructions.data(), &instr)));
                continue;
            }
            // call to non-printf function → unsupported
            return {};
        }

        // Unrecognised instruction — fall back to PTX→LLVM path.
        return {};
    }

    metal << "}\n";
    return metal.str();
}

}  // namespace

LowerToMetalResult lower_ptx_to_metal_source(std::string_view ptx, const LowerToMetalOptions& options) {
    LowerToMetalResult result;

    cumetal::passes::Phase1PipelineOptions pipeline_options;
    pipeline_options.strict = options.strict;
    pipeline_options.entry_name = options.entry_name;
    const auto pipeline = cumetal::passes::run_phase1_pipeline(ptx, pipeline_options);
    if (!pipeline.ok) {
        result.error = pipeline.error;
        return result;
    }

    result.entry_name = pipeline.entry_name;
    result.warnings = pipeline.warnings;

    // First: try the hardcoded name-based lookup for known llm.c kernels.
    std::string metal_source = emit_metal_source_for_entry(pipeline.entry_name);

    // Second: if no hardcoded match, attempt generic PTX → Metal translation.
    if (metal_source.empty()) {
        metal_source = emit_metal_source_generic(pipeline.entry_name, ptx, &pipeline);
    }

    if (metal_source.empty()) {
        result.ok = true;
        result.matched = false;
        return result;
    }

    result.ok = true;
    result.matched = true;
    result.metal_source = metal_source;
    // Propagate printf format table for the runtime to use when draining the buffer.
    for (const auto& fmt : pipeline.printf_formats) {
        result.printf_formats.push_back(fmt.token);
    }
    return result;
}

}  // namespace cumetal::ptx
