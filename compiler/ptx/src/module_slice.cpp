// Reduce a PTX module to what one kernel entry needs.
//
// A translation unit compiles to a single PTX module holding every kernel and
// device function it instantiates; COLMAP's patch_match_cuda.cu is 63 MB with
// 208 entries. The JIT lowers one entry at a time, and every stage after this
// one (parse, pointer inference, both lowering backends, the FP64 provenance
// probe) is at least linear in the text it is handed. Slicing first makes the
// cost of a launch proportional to the kernel rather than to the module.
//
// The slice keeps, in source order:
//   * every module-scope statement that is not a function definition
//     (.version/.target/.address_size, globals, constants, shared and extern
//     declarations, function prototypes) -- these are cheap and the lowering
//     may need any of them;
//   * the requested .entry definition;
//   * every definition reachable from those by name. Reachability is by
//     identifier, not by `call` opcode, so a function whose address is taken
//     (mov.u64 %rd, fn; a vtable initializer) is kept as well, and so is a
//     child .entry that a parent enqueues by symbol (device-side launch).
// Unreferenced entries and unreachable functions are dropped. Nothing is rewritten,
// so the slice is a valid PTX module whenever the input is.

#include "cumetal/ptx/parser.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cumetal::ptx {
namespace {

bool is_ident_start(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_' || c == '$';
}

bool is_ident_char(char c) {
    return is_ident_start(c) || (c >= '0' && c <= '9') || c == '.';
}

struct Item {
    std::size_t begin = 0;
    std::size_t end = 0;      // one past the last byte
    bool definition = false;  // .entry/.func with a body
    bool entry = false;
    std::string name;         // defined symbol, for definitions
};

// Skip a comment or string starting at i; returns the index after it, or i if
// there is none. PTX comments are C-style; `.file` directives carry quoted paths.
std::size_t skip_trivia(std::string_view text, std::size_t i) {
    if (text[i] == '/' && i + 1 < text.size()) {
        if (text[i + 1] == '/') {
            const std::size_t eol = text.find('\n', i);
            return eol == std::string_view::npos ? text.size() : eol;
        }
        if (text[i + 1] == '*') {
            const std::size_t close = text.find("*/", i + 2);
            return close == std::string_view::npos ? text.size() : close + 2;
        }
    }
    if (text[i] == '"') {
        std::size_t j = i + 1;
        while (j < text.size() && text[j] != '"') j += text[j] == '\\' ? 2 : 1;
        return j < text.size() ? j + 1 : text.size();
    }
    return i;
}

// Name defined by a header such as `.weak .func (.param .b32 r) name(` or
// `.visible .entry name(`; empty if the header defines nothing.
std::string defined_name(std::string_view header, bool* is_entry) {
    std::size_t pos = std::string_view::npos;
    std::size_t keyword_len = 0;
    for (std::string_view keyword : {std::string_view(".entry"), std::string_view(".func")}) {
        std::size_t at = 0;
        while ((at = header.find(keyword, at)) != std::string_view::npos) {
            const std::size_t after = at + keyword.size();
            const bool bounded = (at == 0 || !is_ident_char(header[at - 1])) &&
                                 (after >= header.size() || !is_ident_char(header[after]));
            if (bounded) break;
            at = after;
        }
        if (at != std::string_view::npos) {
            pos = at;
            keyword_len = keyword.size();
            *is_entry = keyword == ".entry";
            break;
        }
    }
    if (pos == std::string_view::npos) return {};
    std::size_t i = pos + keyword_len;
    auto skip_space = [&] {
        while (i < header.size() && (header[i] == ' ' || header[i] == '\t' ||
                                     header[i] == '\n' || header[i] == '\r'))
            ++i;
    };
    skip_space();
    if (i < header.size() && header[i] == '(') {  // .func return parameters
        int depth = 0;
        for (; i < header.size(); ++i) {
            if (header[i] == '(') ++depth;
            if (header[i] == ')' && --depth == 0) {
                ++i;
                break;
            }
        }
        skip_space();
    }
    const std::size_t start = i;
    while (i < header.size() && is_ident_char(header[i])) ++i;
    return std::string(header.substr(start, i - start));
}

// Split the module into top-level statements. A statement ends at
//   * the newline after a line directive that takes no `;` (.version,
//     .target, .address_size, .file);
//   * a `;` at brace depth 0 (declarations, including `= { ... };`
//     initializers);
//   * the `}` that closes a function body or any other non-initializer brace
//     group (a debug `.section`).
bool split_items(std::string_view text, std::vector<Item>* items) {
    std::size_t i = 0;
    std::size_t item_begin = 0;
    int depth = 0;
    bool started = false;      // the current statement has a non-trivia token
    bool have_header = false;  // its first top-level `{` has been seen
    bool initializer = false;
    bool definition = false;
    bool entry = false;
    std::string name;
    auto finish = [&](std::size_t end, bool is_definition) {
        items->push_back({item_begin, end, is_definition, is_definition && entry,
                          is_definition ? name : std::string()});
        item_begin = end;
        started = have_header = initializer = definition = entry = false;
        name.clear();
    };
    while (i < text.size()) {
        const std::size_t skipped = skip_trivia(text, i);
        if (skipped != i) {
            i = skipped;
            continue;
        }
        const char c = text[i];
        if (depth == 0 && !started && c == '.') {
            bool line_directive = false;
            for (std::string_view directive : {std::string_view(".version"), std::string_view(".target"),
                                               std::string_view(".address_size"), std::string_view(".file")}) {
                const std::size_t after = i + directive.size();
                if (text.compare(i, directive.size(), directive) == 0 &&
                    (after >= text.size() || !is_ident_char(text[after]))) {
                    std::size_t eol = text.find('\n', i);
                    eol = eol == std::string_view::npos ? text.size() : eol + 1;
                    finish(eol, false);
                    i = eol;
                    line_directive = true;
                    break;
                }
            }
            if (line_directive) continue;
        }
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') started = true;
        if (c == '{') {
            if (depth == 0 && !have_header) {
                have_header = true;
                // The statement's text minus comments and strings: a comment
                // before a definition may itself mention `.entry name(`.
                std::string header;
                for (std::size_t h = item_begin; h < i;) {
                    const std::size_t past = skip_trivia(text, h);
                    if (past != h) {
                        header.push_back(' ');
                        h = past;
                    } else {
                        header.push_back(text[h++]);
                    }
                }
                // An initializer brace follows `=`; a body brace does not.
                const std::size_t eq = header.rfind('=');
                const std::size_t close_paren = header.rfind(')');
                initializer = eq != std::string::npos &&
                    (close_paren == std::string::npos || eq > close_paren);
                if (!initializer) {
                    name = defined_name(header, &entry);
                    definition = !name.empty();
                }
            }
            ++depth;
        } else if (c == '}') {
            if (depth == 0) return false;
            --depth;
            if (depth == 0 && !initializer) {
                finish(i + 1, definition);
                ++i;
                continue;
            }
        } else if (c == ';' && depth == 0) {
            finish(i + 1, false);
            ++i;
            continue;
        }
        ++i;
    }
    if (depth != 0) return false;
    if (item_begin < text.size()) items->push_back({item_begin, text.size(), false, false, {}});
    return true;
}

}  // namespace

std::string slice_module_for_entry(std::string_view text, std::string_view entry_name) {
    std::vector<Item> items;
    if (entry_name.empty() || !split_items(text, &items)) return {};

    // Every definition is reachable by name, entries included: a kernel that
    // enqueues a child kernel (device-side launch) names the child's entry.
    std::unordered_map<std::string_view, std::size_t> functions;
    std::size_t entry_index = items.size();
    for (std::size_t k = 0; k < items.size(); ++k) {
        if (!items[k].definition) continue;
        if (items[k].entry && items[k].name == entry_name) entry_index = k;
        functions.emplace(items[k].name, k);
    }
    if (entry_index == items.size()) return {};

    std::vector<bool> keep(items.size(), false);
    std::vector<std::size_t> work;
    for (std::size_t k = 0; k < items.size(); ++k) {
        if (!items[k].definition || k == entry_index) {
            keep[k] = true;
            work.push_back(k);
        }
    }
    while (!work.empty()) {
        const Item& item = items[work.back()];
        work.pop_back();
        const std::string_view body = text.substr(item.begin, item.end - item.begin);
        for (std::size_t i = 0; i < body.size();) {
            const std::size_t skipped = skip_trivia(body, i);
            if (skipped != i) {
                i = skipped;
                continue;
            }
            if (!is_ident_start(body[i]) || (i > 0 && (is_ident_char(body[i - 1]) || body[i - 1] == '%'))) {
                ++i;
                continue;
            }
            const std::size_t start = i;
            while (i < body.size() && is_ident_char(body[i])) ++i;
            const auto found = functions.find(body.substr(start, i - start));
            if (found != functions.end() && !keep[found->second]) {
                keep[found->second] = true;
                work.push_back(found->second);
            }
        }
    }

    std::string out;
    std::size_t kept_bytes = 0;
    for (std::size_t k = 0; k < items.size(); ++k)
        if (keep[k]) kept_bytes += items[k].end - items[k].begin + 1;
    out.reserve(kept_bytes);
    for (std::size_t k = 0; k < items.size(); ++k) {
        if (!keep[k]) continue;
        out.append(text.substr(items[k].begin, items[k].end - items[k].begin));
        out.push_back('\n');
    }
    return out;
}

}  // namespace cumetal::ptx
