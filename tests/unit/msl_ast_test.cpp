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
    function.statements.push_back(MslStatement::while_statement(
        MslExpression::literal("false", MslType::boolean()),
        {MslStatement::continue_statement()}));
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
    ok &= expect(printed.source.find("continue;") != std::string::npos,
                 "printer emits structured loop continuation");
    ok &= expect(printed.source.find("// cumetal-provenance: generic_ptx_lowering") !=
                     std::string::npos,
                 "printer emits controlled provenance metadata");
    ok &= expect(MslType::pointer(MslType::uint(8), MslAddressSpace::kDevice) ==
                     MslType::pointer(MslType::uint(8), MslAddressSpace::kDevice),
                 "MSL types compare structurally");
    ok &= expect(
        MslType::pointer(
            MslType::pointer(MslType::uint(8), MslAddressSpace::kDevice),
            MslAddressSpace::kThreadgroup)
                .str() == "device uchar* threadgroup*",
        "nested MSL pointers place each address-space qualifier on its own pointer level");

    MslType coherent_pointer = MslType::pointer(MslType::uint(64), MslAddressSpace::kDevice);
    coherent_pointer.device_coherent = true;
    MslType coherent_reference = MslType::reference(MslType::uint(64), MslAddressSpace::kDevice);
    coherent_reference.device_coherent = true;
    ok &= expect(coherent_pointer.str() == "coherent(device) device ulong*" &&
                     coherent_reference.str() == "coherent(device) device ulong&" &&
                     coherent_pointer == coherent_pointer &&
                     !(coherent_pointer ==
                       MslType::pointer(MslType::uint(64), MslAddressSpace::kDevice)),
                 "device coherence is retained in pointer/reference spelling and type identity");
    MslFunction coherent_function;
    coherent_function.name = "coherent_payload";
    coherent_function.parameters = {{.type = coherent_pointer, .name = "payload"}};
    coherent_function.statements.push_back(MslStatement::variable(
        coherent_pointer, "alias", MslExpression::cast(coherent_pointer,
            MslExpression::identifier("payload", coherent_pointer), true)));
    MslModule coherent_module;
    coherent_module.functions.push_back(coherent_function);
    const auto coherent_printed = print_msl(coherent_module);
    ok &= expect(coherent_printed.ok &&
                     coherent_printed.source.find("coherent(device) device ulong* payload") !=
                         std::string::npos &&
                     coherent_printed.source.find(
                         "reinterpret_cast<coherent(device) device cm_alias_ulong*>(payload)") !=
                         std::string::npos,
                 "the MSL printer preserves device coherence through may-alias pointer casts");

    const auto local_pointer = MslExpression::identifier("local_pointer",
        MslType::pointer(MslType::uint(8), MslAddressSpace::kThread));
    MslModule pointer_module;
    MslFunction pointer_function;
    pointer_function.name = "pointer_bits";
    pointer_function.statements.push_back(MslStatement::variable(MslType::uint(64), "bits",
        MslExpression::bitcast(MslType::uint(64), local_pointer)));
    pointer_module.functions.push_back(pointer_function);
    const auto pointer_printed = print_msl(pointer_module);
    ok &= expect(pointer_printed.ok && pointer_printed.source.find("reinterpret_cast<ulong>(local_pointer)") != std::string::npos,
                 "pointer storage words use an address reinterpretation");
    pointer_module.functions[0].statements = {MslStatement::variable(MslType::uint(), "narrow",
        MslExpression::bitcast(MslType::uint(), local_pointer))};
    ok &= expect(!print_msl(pointer_module).ok, "narrow pointer bitcasts fail explicitly");
    pointer_module.functions[0].statements = {
        MslStatement::private_byte_array("private_record", 800, 16),
        MslStatement::threadgroup_byte_array("shared_record", 48, 8),
        MslStatement::variable(MslType::uint(), "aligned_word", std::nullopt, false, 8)};
    const auto aligned = print_msl(pointer_module);
    ok &= expect(aligned.ok &&
        aligned.source.find("alignas(16) thread uchar private_record[800]") != std::string::npos &&
        aligned.source.find("alignas(8) threadgroup uchar shared_record[48]") != std::string::npos &&
        aligned.source.find("alignas(8) uint aligned_word") != std::string::npos,
        "storage declarations retain their requested alignment");
    pointer_module.functions[0].statements = {MslStatement::private_byte_array("bad", 16, 3)};
    ok &= expect(!print_msl(pointer_module).ok, "non-power-of-two storage alignment is rejected");

    if (!ok) return 1;
    std::cout << "MSL AST tests passed\n";
    return 0;
}
