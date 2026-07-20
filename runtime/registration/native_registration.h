#pragma once

#include "registration.h"

namespace cumetal::native_registration {

bool lookup_kernel(const void* host_function,
                   cumetal::registration::RegisteredKernel* out);
void clear();

}  // namespace cumetal::native_registration
