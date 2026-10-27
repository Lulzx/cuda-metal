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

    // Relocation identity: default FakeBuffer device address == contents.
    {
        cumetal::rt::AllocationTable identity_table;
        std::uintptr_t mem[16]{};
        auto identity_buffer = std::make_shared<FakeBuffer>(&mem[0], sizeof(mem));
        if (!expect(identity_table.pointer_relocation_is_identity(true),
                    "empty table relocation is identity (device)")) {
            return 1;
        }
        if (!expect(identity_table.insert(&mem[0], sizeof(mem),
                                          cumetal::rt::AllocationKind::kDevice, 0,
                                          identity_buffer, &error),
                    "insert identity allocation")) {
            return 1;
        }
        if (!expect(identity_table.pointer_relocation_is_identity(true),
                    "host==device relocation is identity (device)")) {
            return 1;
        }
        if (!expect(identity_table.pointer_relocation_is_identity(false),
                    "host==device relocation is identity (host)")) {
            return 1;
        }
        cumetal::rt::AllocationTable shifted_table;
        auto shifted_buffer = std::make_shared<FakeBuffer>(
            &mem[0], sizeof(mem), reinterpret_cast<std::uintptr_t>(&mem[0]) + 0x100000);
        if (!expect(shifted_table.insert(&mem[0], sizeof(mem),
                                         cumetal::rt::AllocationKind::kDevice, 0,
                                         shifted_buffer, &error),
                    "insert shifted allocation")) {
            return 1;
        }
        if (!expect(!shifted_table.pointer_relocation_is_identity(true),
                    "shifted device address is not identity (device)")) {
            return 1;
        }
        if (!expect(shifted_table.pointer_relocation_is_identity(false),
                    "shifted device address is identity (host)")) {
            return 1;
        }
    }

    // MRU cache must not serve stale results across erase + reinsert.
    {
        cumetal::rt::AllocationTable mru_table;
        std::uintptr_t mem[64]{};
        auto mru_buffer = std::make_shared<FakeBuffer>(&mem[0], sizeof(mem));
        if (!expect(mru_table.insert(&mem[0], sizeof(mem),
                                     cumetal::rt::AllocationKind::kDevice, 0,
                                     mru_buffer, &error),
                    "insert MRU allocation")) {
            return 1;
        }
        cumetal::rt::AllocationTable::ResolvedAllocation mru_resolved;
        void* inner = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(&mem[0]) + 32);
        if (!expect(mru_table.resolve(inner, &mru_resolved) && mru_resolved.offset == 32,
                    "MRU first resolve")) {
            return 1;
        }
        if (!expect(mru_table.resolve(inner, &mru_resolved) && mru_resolved.offset == 32 &&
                        mru_resolved.kind == cumetal::rt::AllocationKind::kDevice,
                    "MRU cached resolve")) {
            return 1;
        }
        if (!expect(mru_table.erase(&mem[0]), "MRU erase")) {
            return 1;
        }
        if (!expect(!mru_table.resolve(inner, &mru_resolved),
                    "MRU invalidated after erase")) {
            return 1;
        }
        if (!expect(mru_table.insert(&mem[0], sizeof(mem),
                                     cumetal::rt::AllocationKind::kHost, 0,
                                     mru_buffer, &error),
                    "MRU reinsert as host")) {
            return 1;
        }
        if (!expect(mru_table.resolve(inner, &mru_resolved) &&
                        mru_resolved.kind == cumetal::rt::AllocationKind::kHost,
                    "MRU sees reinserted kind, not stale cache")) {
            return 1;
        }
    }

    std::printf("PASS: allocation table unit tests\n");
    return 0;
}
