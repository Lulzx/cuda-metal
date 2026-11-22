#include "library_provenance.h"

#include "cumetal_diag.h"

#include <cstdio>
#include <string>

namespace cumetal::rt {
namespace {

thread_local LibraryCallScope* g_current_call = nullptr;

}  // namespace

LibraryCallScope::LibraryCallScope(const char* library, const char* routine)
    : library_(library), routine_(routine) {
    // Only the outermost entry point reports: a routine implemented by calling
    // another public routine is still one library call.
    if (g_current_call != nullptr || !diag_env_truthy("CUMETAL_TRACE_GPU")) return;
    active_ = true;
    g_current_call = this;
}

LibraryCallScope::~LibraryCallScope() {
    if (!active_) return;
    g_current_call = nullptr;
    if (backends_ == 0) return;

    const auto has = [this](LibraryBackend backend) {
        return (backends_ & static_cast<std::uint8_t>(backend)) != 0;
    };
    std::string backend;
    const auto append = [&backend](const char* name) {
        if (!backend.empty()) backend += '+';
        backend += name;
    };
    const bool gpu = has(LibraryBackend::kMps) || has(LibraryBackend::kKernel);
    const bool host =
        has(LibraryBackend::kHost) || (has(LibraryBackend::kHostWait) && !gpu &&
                                       !has(LibraryBackend::kGraphCapture));
    if (has(LibraryBackend::kMps)) append("mps");
    if (has(LibraryBackend::kKernel)) append("cumetal_kernel");
    if (host) append("host_fallback");
    if (has(LibraryBackend::kGraphCapture)) append("graph_capture");
    if (backend.empty()) return;  // only waited: no work ran
    std::fprintf(stderr,
                 "CUMETAL_PROVENANCE event=library_call library=%s routine=%s backend=%s "
                 "host_work=%s\n",
                 library_, routine_, backend.c_str(), host ? "true" : "false");
    std::fflush(stderr);
}

void note_library_backend(LibraryBackend backend) {
    if (g_current_call == nullptr) return;
    g_current_call->backends_ |= static_cast<std::uint8_t>(backend);
}

}  // namespace cumetal::rt
