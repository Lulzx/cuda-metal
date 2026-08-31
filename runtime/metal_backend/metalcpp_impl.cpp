// Apple publishes metal-cpp as a header-only C++ wrapper whose private selector
// tables must be instantiated in exactly one translation unit. VkFFT's public
// Metal backend uses that wrapper; CuMetal's own Metal calls remain in the
// Objective-C++ files under this directory.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
