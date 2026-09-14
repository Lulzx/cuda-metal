#include "ptx_instruction.h"
#include "ptx_text.h"

#include <algorithm>
#include <cctype>

namespace cumetal::ir::detail {

// Element count of a `.v2`/`.v4` memory instruction, or 1.
std::size_t memory_vector_width(std::string_view opcode) {
    if (opcode.find(".v4.") != std::string_view::npos) return 4;
    if (opcode.find(".v2.") != std::string_view::npos) return 2;
    return 1;
}

std::string root_opcode(std::string_view opcode) {
    const std::size_t dot = opcode.find('.');
    return std::string(opcode.substr(0, dot));
}

std::vector<std::string> registers_in(std::string_view input) {
    std::vector<std::string> registers;
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (input[i] != '%') {
            continue;
        }
        std::size_t end = i + 1;
        while (end < input.size()) {
            const unsigned char c = static_cast<unsigned char>(input[end]);
            if (std::isalnum(c) == 0 && c != '_' && c != '.' && c != '$') {
                break;
            }
            ++end;
        }
        if (end > i + 1) {
            registers.emplace_back(input.substr(i, end - i));
            i = end - 1;
        }
    }
    return registers;
}

std::string first_register(std::string_view input) {
    const std::vector<std::string> registers = registers_in(input);
    return registers.empty() ? std::string{} : registers.front();
}

std::vector<std::string> destination_registers(const Instruction& instruction) {
    const std::string root = root_opcode(instruction.opcode);
    if (instruction.opcode == "ptx.label" || instruction.operands.empty() ||
        root == "st" || root == "bra" || root == "bar" || root == "membar" ||
        root == "fence" || root == "ret" || root == "exit" || root == "trap" ||
        root == "call") {
        return {};
    }
    std::vector<std::string> destinations = registers_in(instruction.operands.front());
    const bool tuple_move = root == "mov" &&
                            instruction.opcode.find(".b64") != std::string::npos;
    // `ld.*.v2/.v4 {a, b, ...}, [addr]` defines every register of the tuple.
    const bool vector_load = root == "ld" &&
                             (instruction.opcode.find(".v2.") != std::string::npos ||
                              instruction.opcode.find(".v4.") != std::string::npos);
    if (root != "setp" && root != "shfl" && !tuple_move && !vector_load &&
        destinations.size() > 1) {
        destinations.resize(1);
    }
    return destinations;
}

std::vector<std::string> source_registers(const Instruction& instruction) {
    std::vector<std::string> sources;
    const std::string root = root_opcode(instruction.opcode);
    std::size_t first_source = destination_registers(instruction).empty() ? 0 : 1;
    if (root == "st") {
        first_source = 0;
    }
    for (std::size_t i = first_source; i < instruction.operands.size(); ++i) {
        const std::vector<std::string> found = registers_in(instruction.operands[i]);
        sources.insert(sources.end(), found.begin(), found.end());
    }
    if (!instruction.predicate.empty()) {
        const std::string predicate = first_register(instruction.predicate);
        if (!predicate.empty()) {
            sources.push_back(predicate);
        }
    }
    std::erase_if(sources, [](const std::string& name) {
        return starts_with(name, "%tid.") || starts_with(name, "%ctaid.") ||
               starts_with(name, "%ntid.") || starts_with(name, "%nctaid.") ||
               name == "%laneid" || name == "%warpid" || name == "%smid" ||
               name == "%activemask" || starts_with(name, "%clock");
    });
    return sources;
}


std::vector<std::string> grouped_names(std::string_view operand) {
    std::string contents = trim(operand);
    if (contents.size() >= 2 && contents.front() == '(' && contents.back() == ')') {
        contents = trim(std::string_view(contents).substr(1, contents.size() - 2));
    }
    std::vector<std::string> names;
    std::size_t begin = 0;
    while (begin < contents.size()) {
        const std::size_t comma = contents.find(',', begin);
        const std::size_t end = comma == std::string::npos ? contents.size() : comma;
        const std::string name = trim(std::string_view(contents).substr(begin, end - begin));
        if (!name.empty()) names.push_back(name);
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return names;
}


std::string branch_target(const Instruction& instruction) {
    return instruction.operands.empty() ? std::string{} : trim(instruction.operands.back());
}

bool is_conditional_branch(const Instruction& instruction) {
    return root_opcode(instruction.opcode) == "bra" && !instruction.predicate.empty();
}

bool is_terminating_instruction(const Instruction& instruction) {
    const std::string root = root_opcode(instruction.opcode);
    return root == "bra" || root == "ret" || root == "exit" || root == "trap";
}

std::optional<std::string> direct_call_target(const Instruction& instruction) {
    if (root_opcode(instruction.opcode) != "call") return std::nullopt;
    const bool has_return = instruction.operands.size() == 3;
    if ((!has_return && instruction.operands.size() != 2) ||
        (has_return && grouped_names(instruction.operands.front()).size() != 1)) {
        return std::nullopt;
    }
    return trim(instruction.operands[has_return ? 1 : 0]);
}

std::string parameter_name_from_operand(std::string_view operand) {
    const std::size_t open = operand.find('[');
    const std::size_t close = operand.find(']');
    if (open == std::string_view::npos || close == std::string_view::npos || close <= open + 1) {
        return trim(operand);
    }
    std::string inside = trim(operand.substr(open + 1, close - open - 1));
    const std::size_t offset = inside.find_first_of(" +");
    if (offset != std::string::npos) {
        inside.resize(offset);
    }
    return inside;
}

}  // namespace cumetal::ir::detail
