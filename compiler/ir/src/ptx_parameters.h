#pragma once

#include "cumetal/ptx/parser.h"
#include <cstdint>
#include <optional>

namespace cumetal::ir::detail {

std::optional<std::int64_t> parameter_slot_offset(std::string operand, const std::string& name);
void normalize_vector_parameter_transfers(cumetal::ptx::EntryFunction* function);

}  // namespace cumetal::ir::detail
