#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace cumetal::ptx {

struct Parameter {
    std::string type;
    std::string name;
    bool is_pointer = false;
    std::size_t byte_size = 0;
    std::size_t alignment = 1;
};

struct EntryFunction {
    std::string name;
    std::vector<Parameter> params;
    // `.func` definitions may return one or more values through PTX return
    // parameters. Kernel entries leave this empty.
    std::vector<Parameter> return_params;
    struct Instruction {
        std::string predicate;
        std::string opcode;
        std::vector<std::string> operands;
        int line = 0;
        bool supported = false;
    };
    std::vector<Instruction> instructions;
    // Registers declared inside the body by `.reg` directives. A register may
    // be spelled without the `%` prefix (`.reg .pred p;`); the parser renames
    // such names to a `%`-prefixed synthetic register whose NVPTX prefix
    // encodes the declared width (`%p_cm_p`, `%r_cm_tmp`, `%rd_cm_x`, ...) and
    // records the declared PTX type (`pred`, `b32`, `f32`, ...) here so the
    // typed importer can type them before inference.
    struct RegisterDeclaration {
        std::string name;
        std::string type;
    };
    std::vector<RegisterDeclaration> register_declarations;
};

struct ModuleInfo {
    int version_major = -1;
    int version_minor = -1;
    std::string target;
    std::vector<EntryFunction> entries;
    std::vector<EntryFunction> functions;
};

struct ParseOptions {
    bool strict = false;
};

struct ParseResult {
    bool ok = false;
    ModuleInfo module;
    std::vector<std::string> warnings;
    std::string error;
};

ParseResult parse_ptx(std::string_view text);
ParseResult parse_ptx(std::string_view text, const ParseOptions& options);

// Parses a bare instruction block -- the body of an inline `asm` statement
// after operand substitution -- with the same line parser the module parser
// uses, so braces, `.reg` declarations, predicates and the supported-opcode
// classification apply unchanged. Line numbers start at `start_line`.
EntryFunction parse_instruction_block(std::string_view body, int start_line,
                                      std::vector<std::string>* warnings);

}  // namespace cumetal::ptx
