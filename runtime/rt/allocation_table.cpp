#include "allocation_table.h"

namespace cumetal::rt {

bool AllocationTable::insert(void* base,
                             std::size_t size,
                             AllocationKind kind,
                             unsigned int host_alloc_flags,
                             std::shared_ptr<metal_backend::Buffer> buffer,
                             std::string* error_message,
                             bool alias) {
    if (base == nullptr || size == 0 || buffer == nullptr) {
        if (error_message != nullptr) {
            *error_message = "allocation table insert received invalid arguments";
        }
        return false;
    }

    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(base);

    std::unique_lock lock(mutex_);

    auto next = entries_.lower_bound(address);
    if (next != entries_.end()) {
        const std::uintptr_t next_base = next->first;
        if (address + size > next_base) {
            if (error_message != nullptr) {
                *error_message = "allocation overlaps existing entry";
            }
            return false;
        }
    }

    if (next != entries_.begin()) {
        auto prev = next;
        --prev;
        const std::uintptr_t prev_base = prev->first;
        const std::size_t prev_size = prev->second.size;
        if (prev_base + prev_size > address) {
            if (error_message != nullptr) {
                *error_message = "allocation overlaps previous entry";
            }
            return false;
        }
    }

    entries_[address] = Entry{
        .size = size,
        .kind = kind,
        .host_alloc_flags = host_alloc_flags,
        .buffer = std::move(buffer),
        .alias = alias,
    };
    return true;
}

bool AllocationTable::erase(void* base) {
    if (base == nullptr) {
        return false;
    }

    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(base);
    std::unique_lock lock(mutex_);
    const auto found = entries_.find(address);
    if (found == entries_.end()) {
        return false;
    }

    const auto allocation = found->second.buffer;
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->second.buffer == allocation) {
            it = entries_.erase(it);
        } else {
            ++it;
        }
    }
    return true;
}

bool AllocationTable::resolve(const void* ptr, ResolvedAllocation* resolved) const {
    if (ptr == nullptr || resolved == nullptr) {
        return false;
    }

    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(ptr);

    std::shared_lock lock(mutex_);
    auto upper = entries_.upper_bound(address);
    if (upper == entries_.begin()) {
        return false;
    }

    --upper;
    const std::uintptr_t base = upper->first;
    const Entry& entry = upper->second;

    if (address < base || address >= base + entry.size) {
        return false;
    }

    resolved->buffer = entry.buffer;
    resolved->offset = static_cast<std::size_t>(address - base);
    resolved->remaining_size = entry.size - resolved->offset;
    resolved->kind = entry.kind;
    resolved->host_alloc_flags = entry.host_alloc_flags;
    return true;
}

std::size_t AllocationTable::total_allocated_size() const {
    std::shared_lock lock(mutex_);
    std::size_t total = 0;
    for (const auto& [address, entry] : entries_) {
        (void)address;
        if (!entry.alias) {
            total += entry.size;
        }
    }
    return total;
}

void AllocationTable::clear() {
    std::unique_lock lock(mutex_);
    entries_.clear();
}

}  // namespace cumetal::rt
