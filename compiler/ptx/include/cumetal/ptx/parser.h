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

}  // namespace cumetal::ptx
