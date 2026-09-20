#pragma once

#include "cumetal/common/air_target.h"
#include "cumetal/ptx/parser.h"

#include <string>
#include <vector>

namespace cumetal::passes {

struct MetadataField {
    std::string key;
    std::string value;
};

struct KernelMetadata {
    std::string kernel_name;
    std::vector<MetadataField> fields;
};

struct MetadataOptions {
    // Matches the installed Metal Toolchain; air-lld rejects a mismatch.
    std::string air_version = cumetal::common::detected_air_target().version_string();
    std::string language_version = "4.0";
};

KernelMetadata build_kernel_metadata(const cumetal::ptx::EntryFunction& entry,
                                     const MetadataOptions& options = {});

}  // namespace cumetal::passes
