#pragma once

#include "cumetal/ptx/parser.h"

#include <charconv>
#include <string>
#include <string_view>
#include <unordered_map>

namespace cumetal::ir::detail {

// One immutable function per index. Explicit declarations are indexed once;
// compact ranges are never expanded. Ambiguous/scoped bindings remain unknown.
// The caller charges all new work against its own proof budget.
class RegisterWidths {
  public:
    template <class Charge>
    unsigned get(const std::string& name, const cumetal::ptx::EntryFunction& function, Charge charge) {
        if (const auto found = cache_.find(name); found != cache_.end()) return found->second;
        if (cache_.size() >= 65536) return 0;
        if (!indexed_) {
            indexed_ = true;
            if (function.register_declarations.size() > 65536) return 0;
            for (const auto& declaration : function.register_declarations) {
                if (!charge()) { declarations_.clear(); return 0; }
                const auto [found, inserted] = declarations_.emplace(declaration.name,
                    declaration.function_scope ? integer_width(declaration.type) : 0);
                if (!inserted) found->second = 0;
            }
            valid_ = true;
        }
        if (!valid_) return 0;
        const auto exact = declarations_.find(name);
        bool matched = exact != declarations_.end();
        unsigned width = matched ? exact->second : 0;
        for (const auto& range : function.register_ranges) {
            if (!charge()) return 0;
            if (!name.starts_with(range.prefix)) continue;
            const auto digits = std::string_view(name).substr(range.prefix.size());
            if (digits.empty() || (digits.size() > 1 && digits.front() == '0')) continue;
            std::size_t index = 0;
            const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), index);
            if (parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size() || index >= range.count)
                continue;
            width = !matched && range.function_scope ? integer_width(range.type) : 0;
            matched = true;
        }
        cache_.emplace(name, width);
        return width;
    }

  private:
    static unsigned integer_width(std::string_view type) {
        if (type == "b8" || type == "u8" || type == "s8") return 8;
        if (type == "b16" || type == "u16" || type == "s16") return 16;
        if (type == "b32" || type == "u32" || type == "s32") return 32;
        if (type == "b64" || type == "u64" || type == "s64") return 64;
        return 0;
    }
    bool indexed_ = false, valid_ = false;
    std::unordered_map<std::string, unsigned> declarations_, cache_;
};

} // namespace cumetal::ir::detail
