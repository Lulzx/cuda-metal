#include "ptx_local_zero_guards.h"

#include "cumetal/ir/call_write_effects.h"
#include "ptx_text.h"
#include "ptx_pointer_ranges.h"
#include "ptx_registers.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <limits>
#include <set>

namespace cumetal::ir::detail {
namespace {

unsigned integer_width(std::string_view type) {
    if (type == "b8" || type == "u8" || type == "s8") return 8;
    if (type == "b16" || type == "u16" || type == "s16") return 16;
    if (type == "b32" || type == "u32" || type == "s32") return 32;
    if (type == "b64" || type == "u64" || type == "s64") return 64;
    return 0;
}

unsigned memory_width(const Instruction& instruction) {
    const auto dot = instruction.opcode.rfind('.');
    return dot == std::string::npos ? 0 : integer_width(std::string_view(instruction.opcode).substr(dot + 1));
}

bool zero_guard_comparison(const Instruction& instruction) {
    if (!instruction.predicate.empty() || instruction.operands.size() != 3) return false;
    const auto& opcode = instruction.opcode;
    return opcode == "setp.eq.b64" || opcode == "setp.eq.u64" || opcode == "setp.eq.s64" ||
        opcode == "setp.ne.b64" || opcode == "setp.ne.u64" || opcode == "setp.ne.s64" ||
        opcode == "setp.lt.u64" || opcode == "setp.le.u64" ||
        opcode == "setp.gt.u64" || opcode == "setp.ge.u64";
}

bool eligible_zero_load(const Instruction& instruction) {
    return instruction.predicate.empty() && instruction.opcode.starts_with("ld.") &&
        instruction.operands.size() == 2 && memory_width(instruction) == 64 &&
        instruction.opcode.find(".param.") == std::string::npos &&
        instruction.opcode.find(".global.") == std::string::npos &&
        instruction.opcode.find(".shared.") == std::string::npos &&
        instruction.opcode.find(".const.") == std::string::npos;
}

struct Fact {
    // Scalar bits and symbolic allocation addresses are separate domains.
    bool local = false;
    std::string depot;
    std::uint64_t bits = 0;
    std::int64_t offset = 0;
    bool operator==(const Fact&) const = default;
};

struct Proof {
    const std::vector<RawBlock>& blocks;
    const std::vector<std::unordered_map<std::string, ValueId>>& incoming;
    const std::vector<std::unordered_map<std::string, ValueId>>& outgoing;
    const std::vector<std::map<std::string, ValueId>>& arguments;
    const std::unordered_map<const Instruction*, std::vector<ValueId>>& results;
    const std::unordered_map<std::string, LocalDepot>& depots;
    const cumetal::ptx::EntryFunction& function;
    const Module& module;
    LocalZeroLimits limits;
    LocalZeroResult result;

    struct Definition { const Instruction* instruction; std::string name; };
    using Position = std::pair<std::size_t, std::size_t>;
    std::unordered_map<const Instruction*, Position> positions;
    std::unordered_map<const Instruction*, std::unordered_map<std::string, ValueId>> sources;
    std::unordered_map<ValueId, Definition> definitions;
    std::unordered_map<ValueId, std::vector<ValueId>> joins;
    std::unordered_map<ValueId, std::optional<Fact>> facts;
    std::unordered_set<ValueId> active;
    RegisterWidths widths;
    std::unordered_map<std::string, CallEffectSummary> effects;
    std::vector<ValueId> guard_inputs;
    std::unordered_set<ValueId> guard_values;
    std::unordered_map<ValueId, Type> scalar_types;
    std::unique_ptr<ScalarRanges> scalar_ranges;
    std::unique_ptr<PointerRanges> pointer_ranges;

    bool disjoint_store(const Instruction* store, const Fact& cell, std::uint64_t bytes) {
        const auto parsed = address_operand(store->operands[0]);
        if (!parsed || !bytes) return false;
        const auto value = source(store, parsed->first);
        if (!value) return false;
        if (!scalar_ranges) {
            // This pass precedes pointer/type inference. Describe only the
            // declared 64-bit register containers to the scalar proof; its
            // opcode checks still reject unsupported definitions and narrowing.
            // These internal bit widths never become program pointer types.
            for (const auto& [value, definition] : definitions) {
                if (!charge()) return false;
                if (width(definition.name) == 64) scalar_types[value] = Type::integer(64);
            }
            for (const auto& block : arguments) for (const auto& [name, value] : block) {
                if (!charge()) return false;
                if (width(name) == 64) scalar_types[value] = Type::integer(64);
            }
            scalar_ranges = std::make_unique<ScalarRanges>(blocks, incoming, outgoing,
                arguments, results, scalar_types, ScalarRangeLimits{limits.range_work});
            pointer_ranges = std::make_unique<PointerRanges>(blocks, incoming, outgoing,
                arguments, results,
                [&](ValueId value, const Instruction*) -> std::optional<ExactLocalAddress> {
                    const auto known = fact(value);
                    if (!known || !known->local || !in_bounds(*known, 1)) return std::nullopt;
                    return ExactLocalAddress{known->depot, known->offset, depots.at(known->depot).byte_size};
                }, [&](ValueId value, const Instruction* at) { return scalar_ranges->get(value, at); },
                PointerRangeLimits{limits.range_work});
        }
        return pointer_ranges->disjoint(*value, store, parsed->second,
                                        bytes, cell.depot, cell.offset, *scalar_ranges);
    }

    bool charge(std::size_t count = 1) {
        if (count > limits.work - result.work) {
            result.budget_exhausted = true;
            return false;
        }
        result.work += count;
        return true;
    }

    unsigned width(const std::string& name) {
        return widths.get(name, function, [&] { return charge(); });
    }

    std::optional<ValueId> source(const Instruction* instruction, const std::string& name) const {
        const auto found = sources.find(instruction);
        if (found == sources.end()) return std::nullopt;
        const auto value = found->second.find(name);
        return value == found->second.end() ? std::nullopt : std::optional(value->second);
    }

    bool index() {
        if (incoming.size() != blocks.size() || outgoing.size() != blocks.size() || arguments.size() != blocks.size())
            return false;
        for (std::size_t b = 0; b < blocks.size(); ++b) {
            for (const auto& [name, value] : arguments[b]) {
                // Entry always has a caller first-entry path, even with a backedge.
                if (b == 0) continue;
                for (const auto predecessor : blocks[b].predecessors) {
                    if (!charge() || predecessor >= outgoing.size() || !outgoing[predecessor].contains(name)) return false;
                    joins[value].push_back(outgoing[predecessor].at(name));
                }
            }
            auto environment = incoming[b];
            for (std::size_t i = 0; i < blocks[b].instructions.size(); ++i) {
                if (!charge()) return false;
                const auto* instruction = blocks[b].instructions[i];
                positions[instruction] = {b, i};
                for (const auto& name : source_registers(*instruction)) {
                    if (!charge()) return false;
                    if (const auto found = environment.find(name); found != environment.end())
                        sources[instruction][name] = found->second;
                }
                if (zero_guard_comparison(*instruction)) {
                    for (std::size_t operand = 1; operand < instruction->operands.size(); ++operand)
                        if (const auto value = source(instruction, trim(instruction->operands[operand])))
                            guard_inputs.push_back(*value);
                }
                const auto written = destination_registers(*instruction);
                const auto values = results.find(instruction);
                if (values == results.end()) return false;
                // Call return-slot SSA names are not PTX registers. They cannot
                // supply one of this proof's scalar/address leaves.
                if (root_opcode(instruction->opcode) == "call") continue;
                if (written.size() != values->second.size()) return false;
                for (std::size_t j = 0; j < written.size(); ++j) {
                    environment[written[j]] = values->second[j];
                    definitions.emplace(values->second[j], Definition{instruction, written[j]});
                }
            }
        }
        return true;
    }

    bool index_guard_dependencies() {
        while (!guard_inputs.empty()) {
            if (!charge()) return false;
            const auto value = guard_inputs.back();
            guard_inputs.pop_back();
            if (!guard_values.insert(value).second) continue;
            if (const auto join = joins.find(value); join != joins.end()) {
                if (!charge(join->second.size())) return false;
                guard_inputs.insert(guard_inputs.end(), join->second.begin(), join->second.end());
                continue;
            }
            const auto found = definitions.find(value);
            if (found == definitions.end()) continue;
            const auto* instruction = found->second.instruction;
            const auto& op = instruction->opcode;
            if (!instruction->predicate.empty() ||
                (op != "mov.b64" && op != "mov.u64" && op != "mov.s64" && op != "min.u64" &&
                 op != "add.u64" && op != "sub.u64" && op != "add.s64" && op != "sub.s64")) continue;
            for (std::size_t operand = 1; operand < instruction->operands.size(); ++operand) {
                if (!charge()) return false;
                if (const auto input = source(instruction, trim(instruction->operands[operand])))
                    guard_inputs.push_back(*input);
            }
        }
        return true;
    }

    static bool copy64(const Instruction& instruction) {
        return instruction.predicate.empty() && instruction.operands.size() == 2 &&
            (instruction.opcode == "mov.b64" || instruction.opcode == "mov.u64" || instruction.opcode == "mov.s64") &&
            first_register(instruction.operands[0]) == trim(instruction.operands[0]) &&
            first_register(instruction.operands[1]) == trim(instruction.operands[1]);
    }

    std::optional<Fact> operand(const Instruction* at, std::string text) {
        text = trim(text);
        if (const auto number = integer_literal_bits(text)) return Fact{false, {}, *number, 0};
        if (depots.contains(text)) return Fact{true, text, 0, 0};
        if (first_register(text) != text || width(text) != 64) return std::nullopt;
        const auto value = source(at, text);
        return value ? fact(*value) : std::nullopt;
    }

    std::optional<Fact> leaf(ValueId value) {
        const auto found = definitions.find(value);
        if (found == definitions.end() || width(found->second.name) != 64) return std::nullopt;
        const auto* instruction = found->second.instruction;
        if (!instruction->predicate.empty() || destination_registers(*instruction).size() != 1) return std::nullopt;
        const auto& op = instruction->opcode;
        if ((op == "mov.b64" || op == "mov.u64" || op == "mov.s64") && instruction->operands.size() == 2)
            return operand(instruction, instruction->operands[1]);
        if ((op == "cvta.local.u64" || op == "cvta.to.local.u64") && instruction->operands.size() == 2) {
            const auto address = operand(instruction, instruction->operands[1]);
            return address && address->local ? address : std::nullopt;
        }
        if (op == "or.b64" && instruction->operands.size() == 3) {
            auto a = operand(instruction, instruction->operands[1]);
            auto b = operand(instruction, instruction->operands[2]);
            if (!a || !b) return std::nullopt;
            if (!a->local && b->local) std::swap(a, b);
            if (!a->local || b->local || a->offset < 0 || !depots.contains(a->depot)) return std::nullopt;
            const auto alignment = depots.at(a->depot).alignment;
            // Only bits below a proven allocation alignment may be changed.
            // They belong to the offset, independent of the unknown base bits.
            if (!std::has_single_bit(alignment) || b->bits >= alignment) return std::nullopt;
            a->offset = static_cast<std::int64_t>(static_cast<std::uint64_t>(a->offset) | b->bits);
            return a;
        }
        const bool subtract = op == "sub.u64" || op == "sub.s64";
        if ((!subtract && op != "add.u64" && op != "add.s64") || instruction->operands.size() != 3)
            return std::nullopt;
        auto left = operand(instruction, instruction->operands[1]);
        auto right = operand(instruction, instruction->operands[2]);
        if (!left || !right) return std::nullopt;
        if (!left->local && !right->local)
            return Fact{false, {}, subtract ? left->bits - right->bits : left->bits + right->bits, 0};
        if (!subtract && !left->local && right->local) std::swap(left, right);
        if (!left->local || right->local) return std::nullopt;
        const auto displacement = std::bit_cast<std::int64_t>(right->bits);
        const __int128 offset = static_cast<__int128>(left->offset) +
            (subtract ? -static_cast<__int128>(displacement) : displacement);
        if (offset < INT64_MIN || offset > INT64_MAX) return std::nullopt;
        left->offset = static_cast<std::int64_t>(offset);
        return left;
    }

    // Copy/join SCCs need a real leaf on every component. A backedge cannot
    // invent an address or a zero, but invariant copies may relay an existing
    // fact through loops. Every distinct leaf must agree on the exact fact.
    std::optional<Fact> fact(ValueId value) {
        if (!charge()) return std::nullopt;
        if (const auto cached = facts.find(value); cached != facts.end()) return cached->second;
        if (active.size() >= limits.depth) {
            result.budget_exhausted = true;
            return std::nullopt;
        }
        if (!active.insert(value).second) return std::nullopt;
        std::vector<ValueId> pending{value}, leaves;
        std::unordered_set<ValueId> visited;
        std::unordered_map<ValueId, std::vector<ValueId>> reverse;
        std::optional<Fact> answer;
        bool valid = true;
        while (!pending.empty() && valid) {
            if (!charge()) { valid = false; break; }
            const auto current = pending.back();
            pending.pop_back();
            if (!visited.insert(current).second) continue;
            // A completed fact is an anchored leaf even when its definition
            // is a copy/join. Rewalking that already-proved relay for each
            // alias makes otherwise linear zero-field queries quadratic.
            if (const auto cached = facts.find(current); cached != facts.end()) {
                if (!cached->second || (answer && *answer != *cached->second)) { valid = false; break; }
                answer = cached->second;
                leaves.push_back(current);
                continue;
            }
            std::vector<ValueId> inputs;
            if (joins.contains(current)) inputs = joins.at(current);
            else if (definitions.contains(current)) {
                const auto& definition = definitions.at(current);
                if (width(definition.name) == 64 && copy64(*definition.instruction)) {
                    const auto input = source(definition.instruction, trim(definition.instruction->operands[1]));
                    if (input && width(trim(definition.instruction->operands[1])) == 64) inputs.push_back(*input);
                }
            }
            if (inputs.empty()) {
                const auto known = facts.find(current);
                const auto next = known == facts.end() ? leaf(current) : known->second;
                if (!next || (answer && *answer != *next)) { valid = false; break; }
                answer = next;
                leaves.push_back(current);
            } else {
                for (const auto input : inputs) {
                    if (!charge()) { valid = false; break; }
                    reverse[input].push_back(current);
                    pending.push_back(input);
                }
            }
        }
        std::unordered_set<ValueId> anchored;
        while (!leaves.empty() && valid) {
            if (!charge()) { valid = false; break; }
            const auto current = leaves.back();
            leaves.pop_back();
            if (!anchored.insert(current).second) continue;
            if (reverse.contains(current))
                for (const auto user : reverse.at(current)) leaves.push_back(user);
        }
        active.erase(value);
        if (!valid || anchored.size() != visited.size()) answer.reset();
        facts[value] = answer;
        if (answer) for (const auto member : visited) facts[member] = answer;
        return answer;
    }

    static std::optional<std::pair<std::string, std::int64_t>> address_operand(std::string text) {
        text = trim(text);
        if (text.starts_with('[')) {
            if (!text.ends_with(']')) return std::nullopt;
            text = trim(std::string_view(text).substr(1, text.size() - 2));
        }
        const auto sign = text.find_first_of("+-", 1);
        const auto base = trim(std::string_view(text).substr(0, sign));
        std::int64_t displacement = 0;
        if (sign != std::string::npos) {
            const auto literal = integer_literal_bits(trim(std::string_view(text).substr(sign + 1)));
            if (!literal || *literal > INT64_MAX) return std::nullopt;
            displacement = static_cast<std::int64_t>(*literal) * (text[sign] == '-' ? -1 : 1);
        }
        return std::make_pair(base, displacement);
    }

    std::optional<Fact> address(const Instruction* at, std::string text) {
        const auto parsed = address_operand(std::move(text));
        if (!parsed) return std::nullopt;
        auto result = operand(at, parsed->first);
        if (!result || !result->local) return std::nullopt;
        const __int128 offset = static_cast<__int128>(result->offset) + parsed->second;
        if (offset < INT64_MIN || offset > INT64_MAX) return std::nullopt;
        result->offset = static_cast<std::int64_t>(offset);
        return result;
    }

    bool in_bounds(const Fact& cell, std::uint64_t bytes) const {
        return cell.local && cell.offset >= 0 && depots.contains(cell.depot) && bytes != 0 &&
            static_cast<__int128>(cell.offset) + bytes <= depots.at(cell.depot).byte_size;
    }

    static bool overlaps(const Fact& a, std::uint64_t bytes, const Fact& cell) {
        return a.depot == cell.depot && static_cast<__int128>(a.offset) < static_cast<__int128>(cell.offset) + 8 &&
            static_cast<__int128>(cell.offset) < static_cast<__int128>(a.offset) + bytes;
    }

    bool call_disjoint(const Instruction* call, const Fact& cell) {
        const auto target = direct_call_target(*call);
        if (!target || !positions.contains(call)) return false;
        if (!effects.contains(*target)) {
            CallEffectLimits call_limits;
            call_limits.work = std::min(call_limits.work, limits.work - result.work);
            auto summary = summarize_call_effects(module, *target, call_limits);
            if (!charge(summary.work)) return false;
            if (summary.budget_exhausted) result.budget_exhausted = true;
            effects.emplace(*target, std::move(summary));
        }
        const auto& summary = effects.at(*target);
        if (!summary.complete) return false;
        if (summary.writes.empty()) return true;
        if (call->operands.size() != 2 && call->operands.size() != 3) return false;
        const auto names = grouped_names(call->operands.back());
        const auto [block, before] = positions.at(call);
        for (const auto& write : summary.writes) {
            if (!charge() || write.argument >= names.size() || write.bytes == 0) return false;
            const Instruction* staged = nullptr;
            for (std::size_t i = before; i > 0; --i) {
                if (!charge()) return false;
                const auto* prior = blocks[block].instructions[i - 1];
                if (root_opcode(prior->opcode) == "call") break;
                if (!prior->opcode.starts_with("st.param.") || prior->operands.size() != 2 ||
                    parameter_name_from_operand(prior->operands[0]) != names[write.argument]) continue;
                const auto slot = address_operand(prior->operands[0]);
                if (!prior->predicate.empty() || memory_vector_width(prior->opcode) != 1 || memory_width(*prior) != 64 ||
                    !slot || slot->first != names[write.argument] || slot->second != 0) return false;
                staged = prior;
                break;
            }
            if (!staged) return false;
            auto actual = address(staged, staged->operands[1]);
            if (!actual) return false;
            const __int128 offset = static_cast<__int128>(actual->offset) + write.offset;
            if (offset < INT64_MIN || offset > INT64_MAX) return false;
            actual->offset = static_cast<std::int64_t>(offset);
            if (!in_bounds(*actual, write.bytes) || overlaps(*actual, write.bytes, cell)) return false;
        }
        return true;
    }

    bool zero_before(const Instruction* load, const Fact& cell) {
        std::vector<Position> pending{positions.at(load)}, initialized;
        std::set<Position> visited;
        std::map<Position, std::vector<Position>> reverse;
        while (!pending.empty()) {
            if (!charge()) return false;
            if (visited.size() >= limits.states) {
                result.budget_exhausted = true;
                return false;
            }
            const auto current = pending.back();
            pending.pop_back();
            if (!visited.insert(current).second) continue;
            const auto [block, before] = current;
            bool defined = false;
            for (std::size_t i = before; i > 0; --i) {
                if (!charge()) return false;
                const auto* event = blocks[block].instructions[i - 1];
                const auto root = root_opcode(event->opcode);
                if (root == "call") {
                    if (!call_disjoint(event, cell)) return false;
                    continue;
                }
                if (root == "atom" || root == "red") return false;
                if (root != "st" || event->opcode.starts_with("st.param.")) continue;
                if (event->opcode.find(".global.") != std::string::npos ||
                    event->opcode.find(".shared.") != std::string::npos) continue;
                if (event->operands.size() != 2) return false;
                const auto written = address(event, event->operands[0]);
                const auto bytes = memory_width(*event) / 8;
                const auto lanes = memory_vector_width(event->opcode);
                if (!written || !in_bounds(*written, bytes * lanes)) {
                    if (disjoint_store(event, cell, bytes * lanes)) continue;
                    return false;
                }
                if (!overlaps(*written, bytes * lanes, cell)) continue;
                if (bytes != 8 || written->depot != cell.depot || cell.offset < written->offset ||
                    (cell.offset - written->offset) % 8 != 0) return false;
                const auto lane = static_cast<std::size_t>((cell.offset - written->offset) / 8);
                auto contents = trim(event->operands[1]);
                if (contents.size() >= 2 && contents.front() == '{' && contents.back() == '}')
                    contents = trim(std::string_view(contents).substr(1, contents.size() - 2));
                auto values = grouped_names(contents);
                if (values.size() != lanes || lane >= values.size()) return false;
                const auto value = operand(event, values[lane]);
                if (!value || value->local || value->bits != 0) return false;
                // A guarded zero write preserves an earlier proven zero but
                // cannot supply initialization on its untaken path.
                if (!event->predicate.empty()) continue;
                initialized.push_back(current);
                defined = true;
                break;
            }
            if (defined) continue;
            if (block == 0 || blocks[block].predecessors.empty()) return false;
            for (const auto predecessor : blocks[block].predecessors) {
                if (!charge() || predecessor >= blocks.size()) return false;
                const Position previous{predecessor, blocks[predecessor].instructions.size()};
                reverse[previous].push_back(current);
                pending.push_back(previous);
            }
        }
        std::set<Position> anchored;
        while (!initialized.empty()) {
            if (!charge()) return false;
            const auto current = initialized.back();
            initialized.pop_back();
            if (!anchored.insert(current).second) continue;
            if (reverse.contains(current))
                for (const auto user : reverse.at(current)) initialized.push_back(user);
        }
        return anchored.size() == visited.size();
    }

    LocalZeroResult run() {
        if (!index() || !index_guard_dependencies()) return std::move(result);
        for (const auto& block : blocks) for (const auto* instruction : block.instructions) {
            if (!charge()) break;
            if (!eligible_zero_load(*instruction)) continue;
            const auto& values = results.at(instruction);
            if (std::none_of(values.begin(), values.end(), [&](ValueId value) { return guard_values.contains(value); }))
                continue;
            const auto base = address(instruction, instruction->operands[1]);
            const auto written = destination_registers(*instruction);
            if (!base || written.size() != memory_vector_width(instruction->opcode)) continue;
            for (std::size_t lane = 0; lane < written.size(); ++lane) {
                if (!guard_values.contains(values[lane])) continue;
                if (width(written[lane]) != 64) continue;
                auto cell = *base;
                const __int128 offset = static_cast<__int128>(cell.offset) + lane * 8;
                if (offset > INT64_MAX) continue;
                cell.offset = static_cast<std::int64_t>(offset);
                if (in_bounds(cell, 8) && zero_before(instruction, cell)) result.loads[instruction][lane] = 64;
            }
        }
        if (result.budget_exhausted) result.loads.clear();
        return std::move(result);
    }
};

} // namespace

LocalZeroResult prove_local_zero_loads(
    const std::vector<RawBlock>& blocks,
    const std::vector<std::unordered_map<std::string, ValueId>>& incoming,
    const std::vector<std::unordered_map<std::string, ValueId>>& outgoing,
    const std::vector<std::map<std::string, ValueId>>& arguments,
    const std::unordered_map<const Instruction*, std::vector<ValueId>>& results,
    const std::unordered_map<std::string, LocalDepot>& depots,
    const cumetal::ptx::EntryFunction& function, const Module& module, LocalZeroLimits limits) {
    if (depots.empty()) return {};
    // The depot map belongs to the module, so its presence does not mean this
    // function has any relevant memory or guards. Avoid SSA source copies and
    // predecessor indexing until both necessary instruction kinds are present.
    // This scan and the subsequent proof share the same work limit.
    LocalZeroResult preflight;
    bool has_guard = false, has_load = false;
    for (const auto& block : blocks) for (const auto* instruction : block.instructions) {
        if (preflight.work == limits.work) {
            preflight.budget_exhausted = true;
            return preflight;
        }
        ++preflight.work;
        has_guard |= zero_guard_comparison(*instruction);
        has_load |= eligible_zero_load(*instruction);
        if (has_guard && has_load) {
            Proof proof{blocks, incoming, outgoing, arguments, results, depots, function, module, limits};
            proof.result = std::move(preflight);
            return proof.run();
        }
    }
    return preflight;
}

} // namespace cumetal::ir::detail
