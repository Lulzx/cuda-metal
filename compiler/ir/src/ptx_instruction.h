#pragma once

#include "cumetal/ptx/parser.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cumetal::ir::detail {

using Instruction = cumetal::ptx::EntryFunction::Instruction;

std::size_t memory_vector_width(std::string_view opcode);
std::string root_opcode(std::string_view opcode);
std::vector<std::string> registers_in(std::string_view input);
std::string first_register(std::string_view input);
std::vector<std::string> destination_registers(const Instruction& instruction);
std::vector<std::string> source_registers(const Instruction& instruction);
std::vector<std::string> grouped_names(std::string_view operand);
std::string branch_target(const Instruction& instruction);
bool is_conditional_branch(const Instruction& instruction);
bool is_terminating_instruction(const Instruction& instruction);
std::optional<std::string> direct_call_target(const Instruction& instruction);

}  // namespace cumetal::ir::detail
