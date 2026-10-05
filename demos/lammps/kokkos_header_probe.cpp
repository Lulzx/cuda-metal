// Compile-only reproducer: stock Kokkos CUDA headers must parse before any
// LAMMPS translation unit can compile. This is not a GPU execution test.
#include <Kokkos_Core.hpp>

int main() { return 0; }
