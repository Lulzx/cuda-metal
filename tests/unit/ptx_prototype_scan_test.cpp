#include "cumetal/ir/ptx_importer.h"

#include <iostream>
#include <string>

int main() {
    // Generated CUDA names can be long even when no indirect call is present.
    const std::string name = "kernel_" + std::string(8192, 'x');
    const std::string source =
        ".version 7.1\n.target sm_80\n.address_size 64\n.visible .entry " + name +
        "() { ret; }\n";
    const auto imported = cumetal::ir::import_ptx(source);
    if (!imported.ok || imported.module.functions.size() != 1 ||
        imported.module.functions.front().name != name ||
        !cumetal::ir::verify(imported.module).ok) {
        std::cerr << "FAIL: long kernel name without call prototypes: " << imported.error << '\n';
        return 1;
    }
    std::cout << "PASS: long kernel name imports without call prototypes\n";
    return 0;
}
