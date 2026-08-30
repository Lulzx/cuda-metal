#pragma once

#include "registration.h"

namespace cumetal::native_registration {

bool lookup_kernel(const void* host_function,
                   cumetal::registration::RegisteredKernel* out);
bool lookup_symbol(const void* host_symbol, const void** out_device_symbol,
                   std::size_t* out_size);
void reset_device_state();
void clear();

}  // namespace cumetal::native_registration
