#include "cumetal/passes/phase1_pipeline.h"

#include "cumetal/ptx/parser.h"

#include <string>

namespace cumetal::passes {
namespace {

std::string escape_metadata_value(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char c : value) {
        switch (c) {
            case '\n':
                escaped += "\\n";
                break;
            case '\t':
                escaped += "\\t";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            default:
                escaped.push_back(c);
                break;
        }
    }
    return escaped;
}

}  // namespace

Phase1PipelineOutput run_phase1_pipeline(std::string_view ptx, const Phase1PipelineOptions& options) {
    Phase1PipelineOutput out;

    cumetal::ptx::ParseOptions parse_options;
    parse_options.strict = options.strict;
    const auto parsed = cumetal::ptx::parse_ptx(ptx, parse_options);
    if (!parsed.ok) {
        out.error = "ptx parse failed: " + parsed.error;
        return out;
    }
    out.warnings.insert(out.warnings.end(), parsed.warnings.begin(), parsed.warnings.end());

    const cumetal::ptx::EntryFunction* entry = nullptr;
    if (!options.entry_name.empty()) {
        for (const auto& candidate : parsed.module.entries) {
            if (candidate.name == options.entry_name) {
                entry = &candidate;
                break;
            }
        }
        if (entry == nullptr) {
            out.error = "entry not found: " + options.entry_name;
            return out;
        }
    } else {
        entry = &parsed.module.entries.front();
    }
    out.entry_name = entry->name;

    IntrinsicLowerOptions intrinsic_options;
    intrinsic_options.strict = options.strict;
    const auto lowered = lower_intrinsics(*entry, intrinsic_options);
    if (!lowered.ok) {
        out.error = "intrinsic_lower failed: " + lowered.error;
        return out;
    }
    out.lowered_instructions = lowered.instructions;
    out.warnings.insert(out.warnings.end(), lowered.warnings.begin(), lowered.warnings.end());

    PrintfLowerOptions printf_options;
    printf_options.strict = options.strict;
    printf_options.ptx_source = ptx;
    const auto printf_lowered = lower_printf_calls(*entry, printf_options);
    if (!printf_lowered.ok) {
        out.error = "printf_lower failed: " + printf_lowered.error;
        return out;
    }
    out.printf_calls = printf_lowered.calls;
    out.printf_formats = printf_lowered.formats;
    out.warnings.insert(out.warnings.end(), printf_lowered.warnings.begin(), printf_lowered.warnings.end());

    AddrspaceRewriteOptions addrspace_options;
    addrspace_options.strict = options.strict;
    const auto addrspace = rewrite_addrspace(*entry, addrspace_options);
    if (!addrspace.ok) {
        out.error = "addrspace rewrite failed: " + addrspace.error;
        return out;
    }
    out.addrspace_instructions = addrspace.instructions;
    out.warnings.insert(out.warnings.end(), addrspace.warnings.begin(), addrspace.warnings.end());

    out.metadata = build_kernel_metadata(*entry, options.metadata);
    out.metadata.fields.push_back(
        {.key = "kernel.printf.count", .value = std::to_string(out.printf_formats.size())});
    for (const auto& format : out.printf_formats) {
        const std::string prefix = "kernel.printf." + std::to_string(format.id);
        out.metadata.fields.push_back(
            {.key = prefix + ".token", .value = escape_metadata_value(format.token)});
        out.metadata.fields.push_back(
            {.key = prefix + ".literal", .value = format.literal ? "true" : "false"});
        out.metadata.fields.push_back(
            {.key = prefix + ".truncated", .value = format.truncated ? "true" : "false"});
    }
    out.ok = true;
    return out;
}

}  // namespace cumetal::passes
