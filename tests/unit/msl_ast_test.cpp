#include "cumetal/metal/msl_ast.h"

#include <iostream>
#include <string>

namespace {

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    using namespace cumetal::metal;
    bool ok = true;

    const MslExpr a = MslExpression::identifier("a", MslType::uint());
    const MslExpr b = MslExpression::identifier("b", MslType::uint());
    const MslExpr c = MslExpression::identifier("c", MslType::uint());
    const MslExpr expression = MslExpression::binary(
        "*", MslExpression::binary("+", a, b, MslType::uint()), c, MslType::uint());

    MslFunction function;
    function.name = "test.kernel";
    function.is_kernel = true;
    function.parameters = {
        {.type = MslType::uint(), .name = "thread",
         .attributes = {{.name = "thread_index_in_simdgroup"}}},
    };
    function.statements.push_back(
        MslStatement::variable(MslType::uint(), "result", expression, true));
    function.statements.push_back(MslStatement::return_statement());
    MslModule module;
    module.comments.push_back("cumetal-provenance: generic_ptx_lowering");
    module.functions.push_back(function);

    const MslPrintResult printed = print_msl(module);
    ok &= expect(printed.ok, "typed MSL module prints");
    ok &= expect(printed.source.find("(a + b) * c") != std::string::npos,
                 "printer preserves expression precedence");
    ok &= expect(printed.source.find("test_kernel") != std::string::npos,
                 "printer sanitizes function identifiers");
    ok &= expect(printed.source.find("cm_thread") != std::string::npos,
                 "printer protects MSL reserved identifiers");
    ok &= expect(printed.source.find("// cumetal-provenance: generic_ptx_lowering") !=
                     std::string::npos,
                 "printer emits controlled provenance metadata");
    ok &= expect(MslType::pointer(MslType::uint(8), MslAddressSpace::kDevice) ==
                     MslType::pointer(MslType::uint(8), MslAddressSpace::kDevice),
                 "MSL types compare structurally");

    if (!ok) return 1;
    std::cout << "MSL AST tests passed\n";
    return 0;
}
