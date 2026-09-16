#include "ptx_local_memory_ranges.h"
#include "ptx_text.h"

#include <algorithm>
#include <bit>
#include <compare>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <set>

namespace cumetal::ir::detail {
namespace {

enum class Kind { scalar, address, predicate };
struct Fact {
    Kind kind = Kind::scalar;
    std::uint64_t bits = 0;
    unsigned width = 64;
    std::string depot;
    std::int64_t offset = 0;
    auto operator<=>(const Fact&) const = default;
};
using Bytes = std::map<std::pair<std::string, std::int64_t>, std::uint8_t>;
struct State {
    std::size_t block = 0, position = 0;
    std::map<std::string, Fact> registers;
    Bytes bytes;
    auto operator<=>(const State&) const = default;
};
struct MemoryShape {
    unsigned width = 0, lanes = 1;
    bool local = false, external = false;
};

std::optional<std::uint64_t> integer(std::string text) {
    text = trim(text);
    if (text.empty())
        return std::nullopt;
    // PTX integer suffixes are syntax, not part of the value. Floating literals
    // and expressions remain unknown. Negative literals use explicit mod-2^64
    // bits, so -1 is never confused with a positive signed offset.
    if (text.back() == 'U' || text.back() == 'u')
        text.pop_back();
    try {
        std::size_t consumed = 0;
        if (!text.empty() && text.front() == '-') {
            const auto magnitude = std::stoull(text.substr(1), &consumed, 0);
            if (consumed + 1 != text.size())
                return std::nullopt;
            return std::uint64_t{0} - magnitude;
        }
        const auto value = std::stoull(text, &consumed, 0);
        if (consumed == text.size())
            return value;
    } catch (...) {
    }
    return std::nullopt;
}
std::uint64_t mask(unsigned width) { return width == 64 ? UINT64_MAX : (std::uint64_t{1} << width) - 1; }
unsigned scalar_width(const std::string& token) {
    for (const unsigned width : {8U, 16U, 32U, 64U})
        if (token == "b" + std::to_string(width) || token == "u" + std::to_string(width) ||
            token == "s" + std::to_string(width))
            return width;
    return 0;
}
std::optional<MemoryShape> memory_shape(const std::string& opcode) {
    MemoryShape result;
    auto offset = opcode.find('.');
    if (offset == std::string::npos)
        return std::nullopt;
    while (offset < opcode.size()) {
        const auto next = opcode.find('.', offset + 1);
        const auto token = opcode.substr(offset + 1, next - offset - 1);
        if (const auto width = scalar_width(token)) {
            if (result.width)
                return std::nullopt;
            result.width = width;
        } else if (token == "local")
            result.local = true;
        else if (token == "global" || token == "shared" || token == "const" || token == "param")
            result.external = true;
        else if (token == "v2" || token == "v4")
            result.lanes = token == "v2" ? 2 : 4;
        else
            return std::nullopt;
        if (next == std::string::npos)
            break;
        offset = next;
    }
    return result.width && !(result.local && result.external) ? std::optional(result) : std::nullopt;
}
std::vector<std::string> lanes(const std::string& operand) {
    auto text = trim(operand);
    if (!text.empty() && text.front() == '{' && text.back() == '}') {
        text = text.substr(1, text.size() - 2);
        std::vector<std::string> result;
        std::size_t start = 0;
        for (;;) {
            const auto end = text.find(',', start);
            result.push_back(trim(text.substr(start, end - start)));
            if (end == std::string::npos)
                return result;
            start = end + 1;
        }
    }
    return {text};
}
std::vector<std::string> written_registers(const Instruction& instruction) {
    if (root_opcode(instruction.opcode) == "call" && instruction.operands.size() == 3)
        return registers_in(instruction.operands.front());
    return destination_registers(instruction);
}

struct Analysis {
    const std::vector<RawBlock>& blocks;
    const std::unordered_map<std::string, std::uint64_t>& sizes;
    const std::function<bool(const Instruction&)>& preserves;
    LocalMemoryRangeLimits limits;
    LocalMemoryRangeProof proof;
    std::map<const Instruction*, std::optional<LocalStoreRange>> observations;
    bool exhausted = false;

    void exhaust(const std::string& resource, std::size_t used, std::size_t limit,
                 const std::string& phase = {}) {
        if (exhausted)
            return;
        exhausted = true;
        proof.reason = resource + " budget exhausted (used=" + std::to_string(used) +
                       ", limit=" + std::to_string(limit) + (phase.empty() ? "" : ", phase=" + phase) + ")";
    }
    bool spend(std::size_t amount, const std::string& phase) {
        if (exhausted)
            return false;
        if (amount > limits.operations - proof.operations) {
            const auto requested =
                amount > SIZE_MAX - proof.operations ? SIZE_MAX : proof.operations + amount;
            exhaust("operations", requested, limits.operations, phase);
            return false;
        }
        proof.operations += amount;
        return true;
    }

    std::optional<Fact> value(const State& state, const std::string& operand) const {
        const auto text = trim(operand);
        if (const auto found = state.registers.find(text); found != state.registers.end())
            return found->second;
        if (const auto found = sizes.find(text); found != sizes.end() && found->second <= INT64_MAX)
            return Fact{Kind::address, 0, 64, text, 0};
        if (const auto bits = integer(text))
            return Fact{Kind::scalar, *bits};
        return std::nullopt;
    }
    std::optional<bool> predicate(const State& state, const std::string& operand) const {
        const auto [name, inverse] = normalized_predicate(operand);
        const auto found = state.registers.find(name);
        if (found == state.registers.end() || found->second.kind != Kind::predicate)
            return std::nullopt;
        return static_cast<bool>(found->second.bits) != inverse;
    }
    std::optional<Fact> shifted(Fact address, __int128 displacement) const {
        if (address.kind != Kind::address || !sizes.contains(address.depot))
            return std::nullopt;
        const __int128 offset = static_cast<__int128>(address.offset) + displacement;
        if (offset < 0 || offset > sizes.at(address.depot) || offset > INT64_MAX)
            return std::nullopt;
        address.offset = static_cast<std::int64_t>(offset);
        return address;
    }
    std::optional<Fact> address(const State& state, const std::string& operand) const {
        auto text = trim(operand);
        if (text.size() < 3 || text.front() != '[' || text.back() != ']')
            return std::nullopt;
        text = trim(text.substr(1, text.size() - 2));
        const auto sign = text.find_first_of("+-", 1);
        auto base = value(state, sign == std::string::npos ? text : text.substr(0, sign));
        if (!base || base->kind != Kind::address)
            return std::nullopt;
        if (sign == std::string::npos)
            return base;
        const auto displacement = integer(text.substr(sign + 1));
        if (!displacement || *displacement > INT64_MAX)
            return std::nullopt;
        return shifted(*base, (text[sign] == '-' ? -1 : 1) * static_cast<__int128>(*displacement));
    }
    bool footprint(const Fact& base, unsigned bytes) const {
        return base.kind == Kind::address && sizes.contains(base.depot) && base.offset >= 0 &&
               static_cast<unsigned __int128>(base.offset) + bytes <= sizes.at(base.depot);
    }
    void observe(const Instruction* instruction, const std::optional<Fact>& base) {
        auto [found, inserted] = observations.try_emplace(instruction, std::nullopt);
        if (!base) {
            found->second.reset();
            return;
        }
        if (inserted)
            found->second = LocalStoreRange{base->depot, base->offset, base->offset};
        else if (found->second) {
            if (found->second->depot != base->depot)
                found->second.reset();
            else {
                found->second->lower = std::min(found->second->lower, base->offset);
                found->second->upper = std::max(found->second->upper, base->offset);
            }
        }
    }
    void forget_bytes(State& state, const Fact& base, unsigned count) {
        for (unsigned i = 0; i < count; ++i)
            state.bytes.erase({base.depot, base.offset + i});
    }
    void store(State& state, const Instruction* instruction) {
        const auto shape = memory_shape(instruction->opcode);
        if (!shape || instruction->operands.size() != 2) {
            observe(instruction, std::nullopt);
            state.bytes.clear();
            return;
        }
        if (shape->external) {
            observe(instruction, std::nullopt);
            return;
        }
        const auto base = address(state, instruction->operands[0]);
        const unsigned bytes = shape->width / 8;
        if (!base || !footprint(*base, bytes * shape->lanes)) {
            observe(instruction, std::nullopt);
            state.bytes.clear();
            return;
        }
        observe(instruction, base);
        const auto inputs = lanes(instruction->operands[1]);
        forget_bytes(state, *base, bytes * shape->lanes);
        if (inputs.size() != shape->lanes)
            return;
        for (unsigned lane = 0; lane < shape->lanes; ++lane) {
            const auto source = value(state, inputs[lane]);
            if (!source || source->kind != Kind::scalar || source->width < shape->width)
                continue;
            for (unsigned byte = 0; byte < bytes; ++byte)
                state.bytes[{base->depot, base->offset + lane * bytes + byte}] =
                    static_cast<std::uint8_t>(source->bits >> (byte * 8));
        }
    }
    std::vector<std::optional<Fact>> load(const State& state, const Instruction& instruction,
                                          std::size_t destinations) const {
        std::vector<std::optional<Fact>> result(destinations);
        const auto shape = memory_shape(instruction.opcode);
        if (!shape || shape->external || instruction.operands.size() != 2 || shape->lanes != destinations)
            return result;
        const auto base = address(state, instruction.operands[1]);
        const unsigned bytes = shape->width / 8;
        if (!base || !footprint(*base, bytes * shape->lanes))
            return result;
        for (unsigned lane = 0; lane < shape->lanes; ++lane) {
            std::uint64_t bits = 0;
            bool complete = true;
            for (unsigned byte = 0; byte < bytes; ++byte) {
                const auto found = state.bytes.find({base->depot, base->offset + lane * bytes + byte});
                if (found == state.bytes.end()) {
                    complete = false;
                    break;
                }
                bits |= static_cast<std::uint64_t>(found->second) << (byte * 8);
            }
            if (complete)
                result[lane] = Fact{Kind::scalar, bits, shape->width};
        }
        return result;
    }
    std::optional<Fact> compute(const State& state, const Instruction& instruction) const {
        const auto& op = instruction.opcode;
        const auto& operands = instruction.operands;
        const auto get = [&](std::size_t index) { return value(state, operands[index]); };
        if (operands.size() == 2 && operands[1].find('{') == std::string::npos) {
            if (op == "cvta.local.u64" || op == "cvta.to.local.u64") {
                const auto source = get(1);
                return source && source->kind == Kind::address ? source : std::nullopt;
            }
            if (op.starts_with("mov.")) {
                const auto width = scalar_width(op.substr(4));
                const auto source = get(1);
                if (!width || !source)
                    return std::nullopt;
                if (source->kind == Kind::address)
                    return width == 64 ? source : std::nullopt;
                if (source->kind != Kind::scalar || source->width < width)
                    return std::nullopt;
                return Fact{Kind::scalar, source->bits & mask(width), width};
            }
            if (op == "not.pred") {
                const auto source = predicate(state, operands[1]);
                if (source)
                    return Fact{Kind::predicate, !*source, 1};
            }
        }
        if ((op == "and.pred" || op == "or.pred" || op == "xor.pred") && operands.size() == 3) {
            const auto a = predicate(state, operands[1]), b = predicate(state, operands[2]);
            if (op == "and.pred" && ((a && !*a) || (b && !*b)))
                return Fact{Kind::predicate, 0, 1};
            if (op == "or.pred" && ((a && *a) || (b && *b)))
                return Fact{Kind::predicate, 1, 1};
            if (a && b)
                return Fact{Kind::predicate,
                            op == "and.pred"  ? (*a && *b)
                            : op == "or.pred" ? (*a || *b)
                                              : (*a != *b),
                            1};
            return std::nullopt;
        }
        if ((op == "selp.b64" || op == "selp.u64" || op == "selp.s64") && operands.size() == 4) {
            const auto a = get(1), b = get(2);
            const auto selected = predicate(state, operands[3]);
            const auto valid = [](const std::optional<Fact>& fact) {
                return fact && fact->kind != Kind::predicate && fact->width == 64;
            };
            if (selected)
                return valid(*selected ? a : b) ? (*selected ? a : b) : std::nullopt;
            return valid(a) && a == b ? a : std::nullopt;
        }
        if (operands.size() != 3)
            return std::nullopt;
        const auto a = get(1), b = get(2);
        if (!a || !b)
            return std::nullopt;
        if (op == "add.s64" || op == "add.u64" || op == "sub.s64" || op == "sub.u64" || op == "add.s32" ||
            op == "add.u32" || op == "sub.s32" || op == "sub.u32") {
            const bool subtract = op.starts_with("sub.");
            const unsigned width = op.ends_with("32") ? 32 : 64;
            if (a->width < width || b->width < width)
                return std::nullopt;
            if (width == 64 && a->kind == Kind::address && b->kind == Kind::scalar)
                return shifted(*a, (subtract ? -1 : 1) *
                                       static_cast<__int128>(std::bit_cast<std::int64_t>(b->bits)));
            if (width == 64 && !subtract && b->kind == Kind::address && a->kind == Kind::scalar)
                return shifted(*b, std::bit_cast<std::int64_t>(a->bits));
            if (width == 64 && subtract && a->kind == Kind::address && b->kind == Kind::address &&
                a->depot == b->depot)
                return Fact{Kind::scalar, static_cast<std::uint64_t>(a->offset - b->offset)};
            if (a->kind != Kind::scalar || b->kind != Kind::scalar)
                return std::nullopt;
            const auto a_bits = a->bits & mask(width), b_bits = b->bits & mask(width);
            if (op.find(".s") != std::string::npos) {
                const std::int64_t a_signed =
                    width == 32 ? std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(a_bits))
                                : std::bit_cast<std::int64_t>(a_bits);
                const std::int64_t b_signed =
                    width == 32 ? std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(b_bits))
                                : std::bit_cast<std::int64_t>(b_bits);
                const __int128 result =
                    static_cast<__int128>(a_signed) + (subtract ? -1 : 1) * static_cast<__int128>(b_signed);
                if (result < (width == 32 ? INT32_MIN : INT64_MIN) ||
                    result > (width == 32 ? INT32_MAX : INT64_MAX))
                    return std::nullopt;
                return Fact{Kind::scalar, static_cast<std::uint64_t>(result) & mask(width), width};
            }
            if (subtract && a_bits < b_bits)
                return std::nullopt;
            const unsigned __int128 result =
                subtract ? a_bits - b_bits : static_cast<unsigned __int128>(a_bits) + b_bits;
            if (result > mask(width))
                return std::nullopt;
            return Fact{Kind::scalar, static_cast<std::uint64_t>(result), width};
        }
        if (op == "and.b64" && a->kind == Kind::scalar && b->kind == Kind::scalar && a->width == 64 &&
            b->width == 64)
            return Fact{Kind::scalar, a->bits & b->bits};
        if (op == "shl.b64" || op == "shl.b32" || op == "shr.b64" || op == "shr.b32" || op == "shr.u64" ||
            op == "shr.u32") {
            const unsigned width = op.ends_with("32") ? 32 : 64;
            if (a->kind != Kind::scalar || b->kind != Kind::scalar || a->width < width || b->width < 32)
                return std::nullopt;
            const auto count = static_cast<std::uint32_t>(b->bits);
            const auto bits = a->bits & mask(width);
            const auto shifted = count >= width           ? 0
                                 : op.starts_with("shl.") ? (bits << count) & mask(width)
                                                          : bits >> count;
            return Fact{Kind::scalar, shifted, width};
        }
        if (!op.starts_with("setp."))
            return std::nullopt;
        const auto separator = op.find('.', 5);
        if (separator == std::string::npos)
            return std::nullopt;
        const auto relation = op.substr(5, separator - 5), type = op.substr(separator + 1);
        const unsigned width = scalar_width(type);
        if ((width != 32 && width != 64) || a->width < width || b->width < width)
            return std::nullopt;
        bool less = false, equal = false;
        if (width == 64 && a->kind == Kind::address && b->kind == Kind::address && a->depot == b->depot) {
            // Equality depends only on offsets. Do not invent an address bit
            // pattern or use a signed ordering for a symbolic pointer.
            if (relation != "eq" && relation != "ne")
                return std::nullopt;
            equal = a->offset == b->offset;
        } else if (a->kind == Kind::scalar && b->kind == Kind::scalar) {
            const auto a_bits = a->bits & mask(width), b_bits = b->bits & mask(width);
            equal = a_bits == b_bits;
            if (type == "s32")
                less = std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(a_bits)) <
                       std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(b_bits));
            else if (type == "s64")
                less = std::bit_cast<std::int64_t>(a_bits) < std::bit_cast<std::int64_t>(b_bits);
            else
                less = a_bits < b_bits;
        } else
            return std::nullopt;
        std::optional<bool> result;
        if (relation == "eq")
            result = equal;
        else if (relation == "ne")
            result = !equal;
        else if (relation == "lt" && type.front() != 'b')
            result = less;
        else if (relation == "le" && type.front() != 'b')
            result = less || equal;
        else if (relation == "gt" && type.front() != 'b')
            result = !less && !equal;
        else if (relation == "ge" && type.front() != 'b')
            result = !less;
        return result ? std::optional(Fact{Kind::predicate, *result, 1}) : std::nullopt;
    }
    void execute(State& state, const Instruction* instruction) {
        const auto root = root_opcode(instruction->opcode);
        if (root == "st") {
            store(state, instruction);
            return;
        }
        auto destinations = written_registers(*instruction);
        if (root == "call") {
            if (!preserves || !preserves(*instruction))
                state.bytes.clear();
        }
        std::vector<std::optional<Fact>> facts(destinations.size());
        if (root == "ld")
            facts = load(state, *instruction, destinations.size());
        else if (destinations.size() == 1)
            facts[0] = compute(state, *instruction);
        // Compute from the old environment first: `mov x,x` and `add x,x,1`
        // must not read their erased destination. Unsupported writes erase it.
        for (std::size_t i = 0; i < destinations.size(); ++i) {
            state.registers.erase(destinations[i]);
            if (facts[i])
                state.registers.emplace(destinations[i], *facts[i]);
        }
        if (root == "atom" || root == "red" || root == "cp" || root == "sust" || root == "sured")
            state.bytes.clear();
        else if (destinations.empty() && root != "call" && root != "bar" && root != "membar" &&
                 root != "fence" && root != "nop" && instruction->opcode != "ptx.label" && root != ".pragma")
            state.bytes.clear();
    }
    LocalMemoryRangeProof run(const Instruction* target) {
        if (blocks.empty() || !target || root_opcode(target->opcode) != "ld") {
            proof.reason = "invalid load target";
            return proof;
        }
        std::optional<std::size_t> target_block;
        std::size_t target_position = 0;
        for (std::size_t b = 0; b < blocks.size(); ++b) {
            if (!spend(blocks[b].instructions.size() + 1, "locate target"))
                return proof;
            if (std::find(blocks[b].instructions.begin(), blocks[b].instructions.end(), target) !=
                blocks[b].instructions.end()) {
                if (target_block) {
                    proof.reason = "ambiguous load target";
                    return proof;
                }
                target_block = b;
                target_position = static_cast<std::size_t>(
                    std::find(blocks[b].instructions.begin(), blocks[b].instructions.end(), target) -
                    blocks[b].instructions.begin());
            }
        }
        if (!target_block) {
            proof.reason = "load is absent from CFG";
            return proof;
        }
        std::vector<bool> seen(blocks.size(), false), reaches(blocks.size(), false);
        std::vector<std::size_t> pending = blocks[*target_block].successors;
        while (!pending.empty()) {
            if (!spend(1, "target cycle check"))
                return proof;
            const auto b = pending.back();
            pending.pop_back();
            if (b >= blocks.size()) {
                proof.reason = "invalid CFG edge";
                return proof;
            }
            if (b == *target_block) {
                proof.reason = "load belongs to a CFG cycle";
                return proof;
            }
            if (seen[b])
                continue;
            seen[b] = true;
            for (const auto successor : blocks[b].successors)
                pending.push_back(successor);
        }
        std::vector<std::vector<std::size_t>> predecessors(blocks.size());
        for (std::size_t b = 0; b < blocks.size(); ++b) {
            if (!spend(blocks[b].successors.size() + 1, "CFG predecessors"))
                return proof;
            for (const auto successor : blocks[b].successors) {
                if (successor >= blocks.size()) {
                    proof.reason = "invalid CFG edge";
                    return proof;
                }
                predecessors[successor].push_back(b);
            }
        }
        pending = {*target_block};
        while (!pending.empty()) {
            if (!spend(1, "target reachability"))
                return proof;
            const auto b = pending.back();
            pending.pop_back();
            if (reaches[b])
                continue;
            reaches[b] = true;
            for (const auto predecessor : predecessors[b])
                pending.push_back(predecessor);
        }
        if (!reaches[0]) {
            proof.reason = "load unreachable from entry";
            return proof;
        }

        // This is liveness of the prefix analysis, not a program rewrite. A
        // fact can be dropped only if every remaining path defines that name
        // before reading it. Conditional definitions never kill the old value.
        // The target itself is not interpreted, so its operand reads and all
        // following instructions are outside the prefix.
        using Live = std::set<std::string>;
        std::vector<Live> uses(blocks.size()), kills(blocks.size()), live_in(blocks.size()),
            live_out(blocks.size());
        std::vector<std::size_t> ends(blocks.size());
        std::deque<std::size_t> liveness_queue;
        std::vector<bool> queued(blocks.size(), false);
        for (std::size_t b = 0; b < blocks.size(); ++b) {
            if (!reaches[b])
                continue;
            ends[b] = b == *target_block ? target_position : blocks[b].instructions.size();
            for (std::size_t i = 0; i < ends[b]; ++i) {
                const auto& instruction = *blocks[b].instructions[i];
                const auto read = source_registers(instruction), written = written_registers(instruction);
                if (!spend(1 + read.size() + written.size(), "register liveness summary"))
                    return proof;
                for (const auto& name : read)
                    if (!kills[b].contains(name))
                        uses[b].insert(name);
                if (instruction.predicate.empty())
                    for (const auto& name : written)
                        kills[b].insert(name);
            }
            live_in[b] = uses[b];
            liveness_queue.push_back(b);
            queued[b] = true;
        }
        while (!liveness_queue.empty()) {
            const auto b = liveness_queue.front();
            liveness_queue.pop_front();
            queued[b] = false;
            Live next_out;
            if (b != *target_block) {
                for (const auto successor : blocks[b].successors) {
                    if (!reaches[successor])
                        continue;
                    if (!spend(live_in[successor].size() + 1, "register liveness fixed point"))
                        return proof;
                    next_out.insert(live_in[successor].begin(), live_in[successor].end());
                }
            }
            if (!spend(uses[b].size() + next_out.size() + 1, "register liveness fixed point"))
                return proof;
            auto next_in = uses[b];
            for (const auto& name : next_out)
                if (!kills[b].contains(name))
                    next_in.insert(name);
            live_out[b] = std::move(next_out);
            if (next_in == live_in[b])
                continue;
            live_in[b] = std::move(next_in);
            for (const auto predecessor : predecessors[b])
                if (reaches[predecessor] && !queued[predecessor]) {
                    queued[predecessor] = true;
                    liveness_queue.push_back(predecessor);
                }
        }
        std::map<std::pair<std::size_t, std::size_t>, Live> live_at_position;
        const auto live_at = [&](const State& state) -> const Live* {
            if (state.position == 0)
                return &live_in[state.block];
            const auto key = std::make_pair(state.block, state.position);
            if (const auto found = live_at_position.find(key); found != live_at_position.end())
                return &found->second;
            if (state.position > ends[state.block]) {
                exhausted = true;
                proof.reason = "invalid prefix liveness position";
                return nullptr;
            }
            if (!spend(live_out[state.block].size() + 1, "mid-block register liveness"))
                return nullptr;
            Live live = live_out[state.block];
            for (auto i = ends[state.block]; i > state.position; --i) {
                const auto& instruction = *blocks[state.block].instructions[i - 1];
                const auto read = source_registers(instruction), written = written_registers(instruction);
                if (!spend(1 + read.size() + written.size(), "mid-block register liveness"))
                    return nullptr;
                if (instruction.predicate.empty())
                    for (const auto& name : written)
                        live.erase(name);
                live.insert(read.begin(), read.end());
            }
            return &live_at_position.emplace(key, std::move(live)).first->second;
        };

        struct Node {
            State state;
            std::vector<std::size_t> successors;
        };
        std::vector<Node> nodes;
        std::map<State, std::size_t> identities;
        std::deque<std::size_t> queue;
        std::size_t retained_facts = 0;
        const auto enqueue = [&](State state, std::optional<std::size_t> predecessor) {
            if (exhausted || !reaches[state.block])
                return;
            const auto* live = live_at(state);
            if (!live || !spend(state.registers.size() + 1, "prune register facts"))
                return;
            std::erase_if(state.registers, [&](const auto& fact) { return !live->contains(fact.first); });
            auto found = identities.find(state);
            std::size_t id;
            if (found != identities.end())
                id = found->second;
            else {
                if (nodes.size() >= limits.states) {
                    exhaust("states", nodes.size() + 1, limits.states);
                    return;
                }
                const auto cost = state.registers.size() + state.bytes.size() + 1;
                if (cost > limits.retained_facts - retained_facts) {
                    exhaust("retained facts", retained_facts + cost, limits.retained_facts);
                    return;
                }
                retained_facts += cost;
                id = nodes.size();
                identities.emplace(state, id);
                nodes.push_back({std::move(state), {}});
                queue.push_back(id);
            }
            if (predecessor)
                nodes[*predecessor].successors.push_back(id);
        };
        enqueue(State{}, std::nullopt);
        bool reached_load = false;
        while (!queue.empty() && !exhausted) {
            const auto id = queue.front();
            queue.pop_front();
            State state = nodes[id].state;
            bool terminal = false;
            const auto& block = blocks[state.block];
            for (; state.position < block.instructions.size(); ++state.position) {
                if (!spend(1, "prefix instructions"))
                    break;
                const auto* instruction = block.instructions[state.position];
                if (instruction == target) {
                    reached_load = true;
                    terminal = true;
                    break;
                }
                const auto root = root_opcode(instruction->opcode);
                const auto condition = instruction->predicate.empty()
                                           ? std::optional(true)
                                           : predicate(state, instruction->predicate);
                if (root == "bra") {
                    if (state.position + 1 != block.instructions.size() ||
                        block.successors.size() != (instruction->predicate.empty() ? 1U : 2U)) {
                        proof.reason = "unsupported branch shape";
                        return proof;
                    }
                    for (std::size_t edge = 0; edge < block.successors.size(); ++edge) {
                        if (condition && ((*condition && edge != 0) || (!*condition && edge != 1)))
                            continue;
                        auto next = state;
                        next.block = block.successors[edge];
                        next.position = 0;
                        enqueue(std::move(next), id);
                    }
                    terminal = true;
                    break;
                }
                if (condition && !*condition)
                    continue;
                if (!condition) {
                    auto skipped = state;
                    ++skipped.position;
                    enqueue(std::move(skipped), id);
                }
                if (root == "ret" || root == "exit" || root == "trap") {
                    terminal = true;
                    break;
                }
                if (root == "brx") {
                    proof.reason = "unsupported indirect control flow";
                    return proof;
                }
                execute(state, instruction);
                if (state.bytes.size() > limits.known_bytes_per_state) {
                    exhaust("known bytes per state", state.bytes.size(), limits.known_bytes_per_state);
                    break;
                }
                if (state.registers.size() > limits.register_facts_per_state) {
                    exhaust("register facts per state", state.registers.size(),
                            limits.register_facts_per_state);
                    break;
                }
            }
            if (!terminal && !exhausted) {
                if (block.successors.size() > 1) {
                    proof.reason = "unsupported fallthrough shape";
                    return proof;
                }
                for (const auto successor : block.successors) {
                    state.block = successor;
                    state.position = 0;
                    enqueue(std::move(state), id);
                }
            }
        }
        proof.states = nodes.size();
        if (exhausted)
            return proof;
        // Deduplicating an unchanged state is safe for collecting effects, but
        // does not prove termination of an unknown loop. Reject state cycles;
        // finite induction visits distinct exact states and remains acyclic.
        std::vector<std::size_t> indegree(nodes.size());
        for (const auto& node : nodes) {
            if (!spend(node.successors.size() + 1, "prefix state cycle check"))
                return proof;
            for (const auto next : node.successors)
                ++indegree[next];
        }
        queue.clear();
        for (std::size_t i = 0; i < nodes.size(); ++i)
            if (!indegree[i])
                queue.push_back(i);
        std::size_t removed = 0;
        while (!queue.empty()) {
            const auto id = queue.front();
            queue.pop_front();
            ++removed;
            if (!spend(nodes[id].successors.size() + 1, "prefix state cycle check"))
                return proof;
            for (const auto next : nodes[id].successors)
                if (--indegree[next] == 0)
                    queue.push_back(next);
        }
        if (removed != nodes.size()) {
            proof.reason = "repeated prefix state in a loop";
            return proof;
        }
        if (!reached_load) {
            proof.reason = "no feasible prefix reaches load";
            return proof;
        }
        proof.complete = true;
        for (const auto& [instruction, range] : observations)
            if (range)
                proof.stores.emplace(instruction, *range);
        return proof;
    }
};
} // namespace

LocalMemoryRanges::LocalMemoryRanges(const std::vector<RawBlock>& blocks,
                                     const std::unordered_map<std::string, std::uint64_t>& depot_sizes,
                                     std::function<bool(const Instruction&)> preserves_caller_memory,
                                     LocalMemoryRangeLimits limits)
    : blocks_(blocks), depot_sizes_(depot_sizes),
      preserves_caller_memory_(std::move(preserves_caller_memory)), limits_(limits) {}

LocalMemoryRangeProof LocalMemoryRanges::prove_before(const Instruction* target_load) const {
    return Analysis{blocks_, depot_sizes_, preserves_caller_memory_, limits_}.run(target_load);
}

} // namespace cumetal::ir::detail
