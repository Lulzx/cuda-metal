#pragma once

#include "metal_backend.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

namespace cumetal::rt {

enum class AllocationKind {
    kDevice,
    kHost,
};

class AllocationTable {
public:
    struct ResolvedAllocation {
        std::shared_ptr<metal_backend::Buffer> buffer;
        std::size_t offset = 0;
        std::size_t remaining_size = 0;
        AllocationKind kind = AllocationKind::kDevice;
        unsigned int host_alloc_flags = 0;
    };

    bool insert(void* base,
                std::size_t size,
                AllocationKind kind,
                unsigned int host_alloc_flags,
                std::shared_ptr<metal_backend::Buffer> buffer,
                std::string* error_message,
                bool alias = false);
    bool erase(void* base);
    bool resolve(const void* ptr, ResolvedAllocation* resolved) const;
    // Sorted, non-overlapping [base, end) intervals of the current entries.
    // Bulk scans (pointer relocation in memcpy) take one snapshot and filter
    // every word against it without the lock, calling resolve() only for words
    // that actually fall inside an allocation.
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> snapshot_ranges() const;
    std::size_t total_allocated_size() const;
    void clear();

private:
    struct Entry {
        std::size_t size = 0;
        AllocationKind kind = AllocationKind::kDevice;
        unsigned int host_alloc_flags = 0;
        std::shared_ptr<metal_backend::Buffer> buffer;
        bool alias = false;
    };

    std::map<std::uintptr_t, Entry> entries_;
    mutable std::shared_mutex mutex_;
};

}  // namespace cumetal::rt
