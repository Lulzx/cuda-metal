#include "allocation_table.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

namespace {

class FakeBuffer final : public cumetal::metal_backend::Buffer {
public:
    FakeBuffer(void* base, std::size_t size, std::uintptr_t device_address = 0)
        : base_(base), size_(size), device_address_(device_address) {}

    void* contents() const override {
        return base_;
    }

    std::uintptr_t device_address() const override {
        return device_address_ == 0 ? reinterpret_cast<std::uintptr_t>(base_) : device_address_;
    }

    std::size_t length() const override {
        return size_;
    }

private:
    void* base_;
    std::size_t size_;
    std::uintptr_t device_address_;
};

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    cumetal::rt::AllocationTable table;

    std::uintptr_t memory[256]{};
    void* base = &memory[0];
    auto buffer = std::make_shared<FakeBuffer>(base, sizeof(memory));

    std::string error;
    if (!expect(table.insert(base,
                             sizeof(memory),
                             cumetal::rt::AllocationKind::kDevice,
                             0,
                             buffer,
                             &error),
                "insert base allocation")) {
        return 1;
    }
    if (!expect(table.total_allocated_size() == sizeof(memory), "total allocated tracks insert")) {
        return 1;
    }

    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!expect(table.resolve(base, &resolved), "resolve base pointer")) {
        return 1;
    }
    if (!expect(resolved.offset == 0, "base resolve offset is zero")) {
        return 1;
    }
    if (!expect(resolved.remaining_size == sizeof(memory), "base remaining size")) {
        return 1;
    }
    if (!expect(resolved.kind == cumetal::rt::AllocationKind::kDevice, "base allocation kind")) {
        return 1;
    }

    std::uintptr_t alias_storage[256]{};
    void* alias = &alias_storage[0];
    if (!expect(table.insert(alias,
                             sizeof(memory),
                             cumetal::rt::AllocationKind::kDevice,
                             0,
                             buffer,
                             &error,
                             true),
                "insert device alias")) {
        return 1;
    }
    if (!expect(table.total_allocated_size() == sizeof(memory),
                "alias does not double-count allocation size")) {
        return 1;
    }
    if (!expect(table.resolve(alias, &resolved) && resolved.offset == 0,
                "resolve device alias")) {
        return 1;
    }

    void* offset_ptr = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(base) + 64);
    if (!expect(table.resolve(offset_ptr, &resolved), "resolve offset pointer")) {
        return 1;
    }
    if (!expect(resolved.offset == 64, "offset resolve offset bytes")) {
        return 1;
    }

    auto overlap_buffer = std::make_shared<FakeBuffer>(offset_ptr, 32);
    error.clear();
    if (!expect(!table.insert(offset_ptr,
                              32,
                              cumetal::rt::AllocationKind::kDevice,
                              0,
                              overlap_buffer,
                              &error),
                "reject overlapping insert")) {
        return 1;
    }
    if (!expect(!error.empty(), "overlap insert returns error message")) {
        return 1;
    }

    void* miss_ptr = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(base) + sizeof(memory) + 8);
    if (!expect(!table.resolve(miss_ptr, &resolved), "resolve miss pointer")) {
        return 1;
    }

    if (!expect(table.erase(base), "erase existing allocation")) {
        return 1;
    }
    if (!expect(table.total_allocated_size() == 0, "total allocated tracks erase")) {
        return 1;
    }
    if (!expect(!table.resolve(base, &resolved), "resolve after erase fails")) {
        return 1;
    }
    if (!expect(!table.resolve(alias, &resolved), "erase primary removes alias")) {
        return 1;
    }

    if (!expect(!table.erase(base), "erase missing allocation returns false")) {
        return 1;
    }

    std::printf("PASS: allocation table unit tests\n");
    return 0;
}
