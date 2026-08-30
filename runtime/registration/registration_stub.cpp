#include "registration.h"

namespace cumetal::registration {

bool lookup_registered_kernel(const void* host_function, RegisteredKernel* out) {
    (void)host_function;
    (void)out;
    return false;
}

bool lookup_registered_symbol(const void* host_symbol,
                              const void** out_device_symbol,
                              std::size_t* out_size) {
    (void)host_symbol;
    (void)out_device_symbol;
    (void)out_size;
    return false;
}

void reset_device_state() {}

void clear() {}

}  // namespace cumetal::registration
