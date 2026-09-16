#include "ptx_cfg.h"
#include "ptx_text.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <map>
#include <regex>
#include <unordered_map>

namespace cumetal::ir::detail {
namespace {

struct GuardedPaths {
    std::vector<RawBlock>& raw_blocks;
    Builder& builder;
    std::deque<Instruction>& storage;
    const cumetal::ptx::EntryFunction* function;
    InstructionOrigins* origins;

    bool local_predicate(const std::string& reg) const {
        if (!function) return false;
        for (const auto& declaration : function->register_declarations)
            if (declaration.function_scope && declaration.type == "pred" && declaration.name == reg) return true;
        for (const auto& range : function->register_ranges) {
            if (!range.function_scope || range.type != "pred" || !reg.starts_with(range.prefix)) continue;
            const auto digits = std::string_view(reg).substr(range.prefix.size());
            if (digits.empty() || (digits.size() > 1 && digits.front() == '0')) continue;
            std::size_t index = 0;
            const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), index);
            if (parsed.ec == std::errc{} && parsed.ptr == digits.data() + digits.size() &&
                index < range.count) return true;
        }
        return false;
    }

    void invalidate_call(const Instruction& instruction, std::map<std::string, bool>& facts) const {
        const auto target = direct_call_target(instruction);
        const bool direct = (instruction.opcode == "call" || instruction.opcode == "call.uni") &&
            target && !target->empty() && target->front() != '%' &&
            std::all_of(target->begin(), target->end(), [](unsigned char c) {
                return std::isalnum(c) || c == '_' || c == '$' || c == '.';
            });
        if (!direct) { facts.clear(); return; }
        // Callees cannot address caller-local registers. Explicit return
        // registers are still writes, including predicated call outputs.
        const auto outputs = instruction.operands.size() == 3
            ? registers_in(instruction.operands.front()) : std::vector<std::string>{};
        std::erase_if(facts, [&](const auto& item) {
            return !local_predicate(item.first) ||
                std::find(outputs.begin(), outputs.end(), item.first) != outputs.end();
        });
    }

    std::size_t cloned_blocks = 0;
    std::size_t cloned_instructions = 0;

    // A bounded proof may decline to simplify. Never invent a definition when
    // a proof or budget is exhausted: subsequent SSA construction still checks it.
    std::optional<std::size_t> clone_edge(std::size_t parent, std::size_t edge,
                                         const RawBlock& target, std::size_t successor) {
        const bool has_branch = !target.instructions.empty() &&
            root_opcode(target.instructions.back()->opcode) == "bra";
        const auto count = target.instructions.size() + (has_branch ? 0 : 1);
        if (cloned_blocks >= 4096 || count > 131072 - cloned_instructions) return std::nullopt;
        RawBlock clone;
        clone.id = builder.next_block();
        clone.name = target.name + "_guard_" + std::to_string(raw_blocks.size());
        clone.successors = {successor};
        for (std::size_t j = 0; j < target.instructions.size(); ++j) {
            auto instruction = *target.instructions[j];
            if (has_branch && j + 1 == target.instructions.size()) {
                instruction.predicate.clear();
                instruction.operands = {raw_blocks[successor].name};
            }
            storage.push_back(std::move(instruction));
            record_instruction_origin(origins, &storage.back(), target.instructions[j]);
            clone.instructions.push_back(&storage.back());
        }
        if (!has_branch) {
            Instruction branch;
            branch.opcode = "bra";
            branch.operands = {raw_blocks[successor].name};
            storage.push_back(std::move(branch));
            clone.instructions.push_back(&storage.back());
        }
        const auto index = raw_blocks.size();
        raw_blocks[parent].successors[edge] = index;
        ++cloned_blocks;
        cloned_instructions += clone.instructions.size();
        raw_blocks.push_back(std::move(clone));
        return index;
    }

    struct ThresholdPredicate {
        std::string reg;
        std::string width;
        unsigned long long threshold;
        bool greater_equal;
    };

    static std::optional<ThresholdPredicate> threshold_predicate(const Instruction& instruction) {
        static const std::regex comparison(R"(^setp\.(lt|le|gt|ge)\.u(32|64)$)");
        std::smatch match;
        if (!instruction.predicate.empty() || instruction.operands.size() != 3 ||
            !std::regex_match(instruction.opcode, match, comparison)) return std::nullopt;
        const auto reg = trim(instruction.operands[1]);
        const auto immediate = trim(instruction.operands[2]);
        if (reg.empty() || first_register(reg) != reg || immediate.empty() ||
            immediate.find_first_not_of("0123456789") != std::string::npos) return std::nullopt;
        try {
            auto threshold = std::stoull(immediate);
            const auto maximum = match[2] == "32" ? 0xffffffffULL : ~0ULL;
            const std::string op = match[1];
            if (threshold > maximum || ((op == "gt" || op == "le") && threshold == maximum))
                return std::nullopt;
            if (op == "gt" || op == "le") ++threshold;
            return ThresholdPredicate{reg, match[2], threshold, op == "gt" || op == "ge"};
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }


    // Specialize a compare-only successor on a known incoming comparison edge.
    // Keep its predicate definition, but remove the impossible outgoing edge.
    // This exposes the proof to both self-select liveness and register SSA.
    void thread_threshold_edges() {
        const std::size_t original_count = raw_blocks.size();
        for (std::size_t index = 0; index < original_count; ++index) {
            const RawBlock source = raw_blocks[index];
            if (source.instructions.empty() || source.successors.size() != 2) continue;
            const Instruction* branch = source.instructions.back();
            if (root_opcode(branch->opcode) != "bra" || branch->predicate.empty()) continue;
            const auto [pred, inverted] = normalized_predicate(branch->predicate);
            std::optional<ThresholdPredicate> condition;
            // Remember a comparison only while its predicate and input are unchanged.
            for (std::size_t i = 0; i + 1 < source.instructions.size(); ++i) {
                const auto& instruction = *source.instructions[i];
                const auto written = destination_registers(instruction);
                if (condition && (std::find(written.begin(), written.end(), pred) != written.end() ||
                                  std::find(written.begin(), written.end(), condition->reg) != written.end()))
                    condition.reset();
                if (root_opcode(instruction.opcode) == "call") condition.reset();
                if (written.size() == 1 && written[0] == pred)
                    condition = threshold_predicate(instruction);
            }
            if (!condition) continue;
            for (std::size_t edge = 0; edge < 2; ++edge) {
                const RawBlock target = raw_blocks[source.successors[edge]];
                if (target.instructions.size() != 2 || target.successors.size() != 2) continue;
                const auto next = threshold_predicate(*target.instructions[0]);
                const Instruction* tail = target.instructions[1];
                if (!next || root_opcode(tail->opcode) != "bra" || tail->predicate.empty()) continue;
                const auto [next_pred, next_inverted] = normalized_predicate(tail->predicate);
                const auto written = destination_registers(*target.instructions[0]);
                if (written.size() != 1 || written[0] != next_pred || next->reg != condition->reg ||
                    next->width != condition->width || next->threshold != condition->threshold) continue;
                const bool known = ((edge == 0) != inverted) == condition->greater_equal;
                const bool take = (known == next->greater_equal) != next_inverted;
                clone_edge(index, edge, target, target.successors[take ? 0 : 1]);
            }
        }
    }

    struct ComparisonFact {
        std::string left, right, type, relation;
        bool positive;
        std::vector<std::string> dependencies{};
    };

    static std::optional<ComparisonFact> comparison_predicate(const Instruction& instruction) {
        static const std::regex comparison(R"(^setp\.(eq|ne|lt|le|gt|ge)\.([bu](32|64))$)");
        std::smatch match;
        if (!instruction.predicate.empty() || instruction.operands.size() != 3 ||
            !std::regex_match(instruction.opcode, match, comparison)) return std::nullopt;
        if (first_register(instruction.operands[0]) != trim(instruction.operands[0])) return std::nullopt;
        auto left = trim(instruction.operands[1]);
        auto right = trim(instruction.operands[2]);
        // Restrict the proof to scalar register comparisons; malformed operands
        // and other comparison semantics remain the responsibility of the importer.
        if (left.empty() || right.empty() || first_register(left) != left ||
            first_register(right) != right) return std::nullopt;
        const std::string operation = match[1];
        const bool equality = operation == "eq" || operation == "ne";
        // Canonicalize to equality or unsigned less-than. Operand order matters
        // for inequalities; complements share a fact but invert its polarity.
        if (!equality && match[2].str()[0] != 'u') return std::nullopt;
        if ((equality && right < left) || operation == "gt" || operation == "le")
            std::swap(left, right);
        return ComparisonFact{left, right, match[2], equality ? "eq" : "lt",
                              operation == "eq" || operation == "lt" || operation == "gt"};
    }


    // Track single pure expressions without rewriting their computations.
    // Writes invalidate both aliases and snapshots used by a guard proof.
    struct Expressions {
        struct Value { std::string key; std::vector<std::string> dependencies{}; };
        std::map<std::string, Value> values;
        std::deque<std::string> order;
        void invalidate(const Instruction& instruction) {
            const auto writes = destination_registers(instruction);
            for (const auto& reg : writes) {
                std::erase_if(values, [&](const auto& item) {
                    return item.first == reg || std::find(item.second.dependencies.begin(),
                        item.second.dependencies.end(), reg) != item.second.dependencies.end();
                });
            }
            std::erase_if(order, [&](const auto& reg) { return !values.contains(reg); });
            if (root_opcode(instruction.opcode) == "call") { values.clear(); order.clear(); }
        }
        void observe(const Instruction& instruction) {
            const auto writes = destination_registers(instruction);
            invalidate(instruction);
            if (root_opcode(instruction.opcode) == "call") return;
            if (!instruction.predicate.empty() || writes.size() != 1) return;
            const bool add = instruction.opcode == "add.u32" || instruction.opcode == "add.s32" ||
                instruction.opcode == "add.u64" || instruction.opcode == "add.s64";
            const bool convert = instruction.opcode == "cvt.u64.u32";
            if ((!add && !convert) || instruction.operands.size() != (add ? 3u : 2u)) return;
            const auto destination = trim(instruction.operands[0]);
            const unsigned width = instruction.opcode.ends_with("32") && add ? 32 : 64;
            if (first_register(destination) != destination ||
                ptx_register_container_bits(destination) != width) return;
            Value value{instruction.opcode, {}};
            static const std::regex integer(R"(^-?[0-9]+$)");
            for (std::size_t i = 1; i < instruction.operands.size(); ++i) {
                const auto operand = trim(instruction.operands[i]);
                if (first_register(operand) == operand && !operand.empty() &&
                    ptx_register_container_bits(operand) == (convert ? 32 : width)) {
                    // In-place expressions require a versioned value proof.
                    if (operand == destination) return;
                    value.dependencies.push_back(operand);
                } else if (!std::regex_match(operand, integer)) return;
                value.key += "|" + operand;
            }
            // Forget the oldest expression when the proof reaches its bound.
            // Eviction only loses an alias; it cannot manufacture a value.
            if (values.size() >= 64) { values.erase(order.front()); order.pop_front(); }
            order.push_back(destination);
            values[destination] = std::move(value);
        }
        std::optional<ComparisonFact> comparison(const Instruction& instruction) const {
            auto fact = comparison_predicate(instruction);
            if (!fact) return fact;
            fact->dependencies = {fact->left, fact->right};
            auto key = [&](const std::string& reg) {
                const auto found = values.find(reg);
                if (found == values.end()) return "reg|" + reg;
                fact->dependencies.insert(fact->dependencies.end(), found->second.dependencies.begin(),
                                           found->second.dependencies.end());
                return "expr|" + found->second.key;
            };
            fact->left = key(fact->left);
            fact->right = key(fact->right);
            if (fact->relation == "eq" && fact->right < fact->left) std::swap(fact->left, fact->right);
            return fact;
        }
    };

    // Seed expression identities from a bounded predecessor chain. A
    // single-entry loop may retain only identities untouched in its SCC.
    Expressions expressions_before(std::size_t index) const {
        if (raw_blocks[index].predecessors.empty()) return {};
        std::unordered_set<std::size_t> loop_blocks;
        std::optional<std::size_t> first;
        if (raw_blocks[index].predecessors.size() == 1) {
            first = raw_blocks[index].predecessors.front();
        } else {
            const auto walk = [&](bool backwards) -> std::optional<std::vector<bool>> {
                std::vector<bool> reached(raw_blocks.size(), false);
                std::deque<std::size_t> pending{index};
                std::size_t budget = 131072;
                while (!pending.empty()) {
                    if (budget == 0) return std::nullopt;
                    --budget;
                    const auto block = pending.front();
                    pending.pop_front();
                    if (reached[block]) continue;
                    reached[block] = true;
                    const auto& next = backwards ? raw_blocks[block].predecessors
                                                 : raw_blocks[block].successors;
                    pending.insert(pending.end(), next.begin(), next.end());
                }
                return reached;
            };
            const auto forward = walk(false);
            const auto backward = walk(true);
            if (!forward || !backward) return {};
            for (std::size_t block = 0; block < raw_blocks.size(); ++block)
                if ((*forward)[block] && (*backward)[block]) loop_blocks.insert(block);
            std::unordered_set<std::size_t> external;
            for (const auto predecessor : raw_blocks[index].predecessors)
                if (!loop_blocks.contains(predecessor)) external.insert(predecessor);
            if (external.size() != 1 || loop_blocks.size() < 2) return {};
            first = *external.begin();
            // Irreducible entries can bypass the preheader expression.
            for (const auto block : loop_blocks)
                for (const auto predecessor : raw_blocks[block].predecessors)
                    if (!loop_blocks.contains(predecessor) &&
                        (block != index || predecessor != *first)) return {};
        }

        std::vector<std::size_t> prefix;
        auto visited = loop_blocks;
        visited.insert(index);
        std::size_t instructions = 0;
        auto current = *first;
        for (unsigned depth = 0; depth < 8; ++depth) {
            if (!visited.insert(current).second) return {};
            const auto count = raw_blocks[current].instructions.size();
            if (count > 256 - instructions) break;
            instructions += count;
            prefix.push_back(current);
            if (raw_blocks[current].predecessors.size() != 1) break;
            current = raw_blocks[current].predecessors.front();
        }
        std::reverse(prefix.begin(), prefix.end());
        Expressions expressions;
        for (const auto block : prefix)
            for (const auto* instruction : raw_blocks[block].instructions)
                expressions.observe(*instruction);
        std::size_t loop_budget = 131072;
        for (const auto block : loop_blocks) {
            for (const auto* instruction : raw_blocks[block].instructions) {
                if (loop_budget == 0) return {};
                --loop_budget;
                expressions.invalidate(*instruction);
            }
        }
        return expressions;
    }

    static bool writes_fact(const std::string& reg, const ComparisonFact& fact) {
        return std::find(fact.dependencies.begin(), fact.dependencies.end(), reg) != fact.dependencies.end();
    }

    static std::optional<bool> predicate_value(
        const Instruction& instruction, const std::map<std::string, bool>& known) {
        if (!instruction.predicate.empty()) return std::nullopt;
        auto lookup = [&](const std::string& operand) -> std::optional<bool> {
            const auto text = trim(operand);
            if (text == "0") return false;
            if (text == "1" || text == "-1") return true;
            const auto found = known.find(text);
            return found == known.end() ? std::nullopt : std::optional<bool>(found->second);
        };
        if (instruction.operands.size() == 2 &&
            (instruction.opcode == "mov.pred" || instruction.opcode == "not.pred")) {
            auto value = lookup(instruction.operands[1]);
            if (value && instruction.opcode == "not.pred") value = !*value;
            return value;
        }
        if (instruction.operands.size() == 3 && instruction.opcode == "or.pred") {
            auto a = lookup(instruction.operands[1]);
            auto b = lookup(instruction.operands[2]);
            if ((a && *a) || (b && *b)) return true;
            if (a && b) return false;
        }
        if (instruction.operands.size() == 3 && instruction.opcode == "and.pred") {
            auto a = lookup(instruction.operands[1]);
            auto b = lookup(instruction.operands[2]);
            if ((a && !*a) || (b && !*b)) return false;
            if (a && b) return true;
        }
        return std::nullopt;
    }

    struct PredicateLiveness {
        using Registers = std::unordered_set<std::string>;
        std::vector<Registers> live_in;
        std::vector<Registers> live_out;
        std::vector<std::unordered_map<std::string, std::size_t>> last_use;
    };

    // Constant propagation needs only predicate values that can still be read.
    // Compute that set from instruction semantics rather than register names:
    // these are exactly the destinations the constant evaluator can create.
    std::optional<PredicateLiveness> predicate_liveness() const {
        using Registers = PredicateLiveness::Registers;
        Registers candidates;
        for (const auto& block : raw_blocks) {
            for (const auto* instruction : block.instructions) {
                if (instruction->opcode != "mov.pred" && instruction->opcode != "not.pred" &&
                    instruction->opcode != "or.pred") continue;
                const auto written = destination_registers(*instruction);
                if (written.size() == 1) candidates.insert(written[0]);
            }
        }

        PredicateLiveness result;
        result.live_in.resize(raw_blocks.size());
        result.live_out.resize(raw_blocks.size());
        result.last_use.resize(raw_blocks.size());
        std::vector<Registers> uses(raw_blocks.size()), definitions(raw_blocks.size());
        std::size_t budget = 16777216;
        for (std::size_t b = 0; b < raw_blocks.size(); ++b) {
            for (std::size_t i = 0; i < raw_blocks[b].instructions.size(); ++i) {
                const auto& instruction = *raw_blocks[b].instructions[i];
                for (const auto& source : source_registers(instruction)) {
                    if (!candidates.contains(source)) continue;
                    if (budget == 0) return std::nullopt;
                    --budget;
                    result.last_use[b][source] = i;
                    if (!definitions[b].contains(source)) uses[b].insert(source);
                }
                for (const auto& written : destination_registers(instruction)) {
                    if (candidates.contains(written)) definitions[b].insert(written);
                }
            }
        }

        std::deque<std::size_t> pending;
        std::vector<bool> queued(raw_blocks.size(), true);
        for (std::size_t b = raw_blocks.size(); b-- > 0;) pending.push_back(b);
        std::size_t stored = 0;
        while (!pending.empty()) {
            if (budget == 0) return std::nullopt;
            --budget;
            const auto b = pending.front();
            pending.pop_front();
            queued[b] = false;
            Registers live_out;
            for (const auto successor : raw_blocks[b].successors) {
                if (result.live_in[successor].size() > budget) return std::nullopt;
                budget -= result.live_in[successor].size();
                live_out.insert(result.live_in[successor].begin(), result.live_in[successor].end());
            }
            Registers live_in = uses[b];
            for (const auto& reg : live_out)
                if (!definitions[b].contains(reg)) live_in.insert(reg);
            if (live_in == result.live_in[b] && live_out == result.live_out[b]) continue;
            stored -= result.live_in[b].size() + result.live_out[b].size();
            stored += live_in.size() + live_out.size();
            if (stored > 1048576) return std::nullopt;
            result.live_in[b] = std::move(live_in);
            result.live_out[b] = std::move(live_out);
            for (const auto predecessor : raw_blocks[b].predecessors) {
                if (!queued[predecessor]) {
                    pending.push_back(predecessor);
                    queued[predecessor] = true;
                }
            }
        }
        return result;
    }

    // Reachable predecessor states meet by intersection. An unvisited edge is
    // not an unknown value: ignoring it until visited preserves loop-invariant
    // flags, while later conflicting edges remove the fact and requeue users.
    std::vector<std::map<std::string, bool>> constant_entries() const {
        using Facts = std::map<std::string, bool>;
        std::vector<Facts> entries(raw_blocks.size());
        if (raw_blocks.empty()) return entries;
        const auto liveness = predicate_liveness();
        if (!liveness) return entries;
        std::vector<bool> visited(raw_blocks.size(), false), queued(raw_blocks.size(), false);
        std::deque<std::size_t> pending{0};
        visited[0] = queued[0] = true;
        std::size_t steps = 0, instructions = 0;
        while (!pending.empty()) {
            if (++steps > 131072) return std::vector<Facts>(raw_blocks.size());
            const auto index = pending.front();
            pending.pop_front();
            queued[index] = false;
            auto outgoing = entries[index];
            for (std::size_t i = 0; i < raw_blocks[index].instructions.size(); ++i) {
                const auto* instruction = raw_blocks[index].instructions[i];
                if (++instructions > 4194304) return std::vector<Facts>(raw_blocks.size());
                const auto value = predicate_value(*instruction, outgoing);
                const auto written = destination_registers(*instruction);
                for (const auto& reg : written) outgoing.erase(reg);
                if (root_opcode(instruction->opcode) == "call") invalidate_call(*instruction, outgoing);
                if (value && written.size() == 1) outgoing[written[0]] = *value;
                std::erase_if(outgoing, [&](const auto& item) {
                    const auto last = liveness->last_use[index].find(item.first);
                    return !liveness->live_out[index].contains(item.first) &&
                        (last == liveness->last_use[index].end() || last->second <= i);
                });
                if (outgoing.size() > 128) return std::vector<Facts>(raw_blocks.size());
            }
            for (const auto successor : raw_blocks[index].successors) {
                auto transferred = outgoing;
                std::erase_if(transferred, [&](const auto& item) {
                    return !liveness->live_in[successor].contains(item.first);
                });
                auto merged = transferred;
                if (visited[successor]) {
                    merged = entries[successor];
                    std::erase_if(merged, [&](const auto& item) {
                        const auto found = transferred.find(item.first);
                        return found == transferred.end() || found->second != item.second;
                    });
                }
                if (visited[successor] && merged == entries[successor]) continue;
                visited[successor] = true;
                entries[successor] = std::move(merged);
                if (!queued[successor]) {
                    pending.push_back(successor);
                    queued[successor] = true;
                }
            }
        }
        return entries;
    }

    // Generated Option equality first computes an otherwise undefined payload,
    // then absorbs its comparison with a false presence predicate. Retain the
    // alternatives explicitly until the predicate is no longer live.
    enum class MaskState : unsigned char { Unknown, False, True };

    static unsigned state_bit(MaskState state) {
        return 1u << static_cast<unsigned>(state);
    }

    MaskState transfer_mask(const RawBlock& block, const std::string& mask,
                            MaskState state) const {
        for (const auto* instruction : block.instructions) {
            if (root_opcode(instruction->opcode) == "call" && state != MaskState::Unknown) {
                std::map<std::string, bool> facts{{mask, state == MaskState::True}};
                invalidate_call(*instruction, facts);
                if (!facts.contains(mask)) state = MaskState::Unknown;
            }
            const auto written = destination_registers(*instruction);
            if (std::find(written.begin(), written.end(), mask) == written.end()) continue;
            std::map<std::string, bool> known;
            if (state != MaskState::Unknown) known[mask] = state == MaskState::True;
            const auto value = predicate_value(*instruction, known);
            state = value ? (*value ? MaskState::True : MaskState::False)
                          : MaskState::Unknown;
        }
        return state;
    }

    std::optional<MaskState> transfer_mask_edge(const RawBlock& block,
                                                const std::string& mask,
                                                MaskState state,
                                                std::size_t edge) const {
        state = transfer_mask(block, mask, state);
        if (block.instructions.empty() || block.successors.size() != 2) return state;
        const auto* tail = block.instructions.back();
        if (root_opcode(tail->opcode) != "bra" || tail->predicate.empty()) return state;
        const auto [predicate, inverted] = normalized_predicate(tail->predicate);
        if (predicate != mask) return state;
        const bool edge_value = (edge == 0) != inverted;
        if (state != MaskState::Unknown && (state == MaskState::True) != edge_value)
            return std::nullopt;
        return edge_value ? MaskState::True : MaskState::False;
    }

    std::optional<std::vector<bool>> mask_liveness(const std::string& mask) const {
        std::vector<bool> uses(raw_blocks.size(), false);
        std::vector<bool> definitions(raw_blocks.size(), false);
        std::size_t budget = 16777216;
        for (std::size_t b = 0; b < raw_blocks.size(); ++b) {
            bool defined = false;
            for (const auto* instruction : raw_blocks[b].instructions) {
                if (budget == 0) return std::nullopt;
                --budget;
                const auto sources = source_registers(*instruction);
                if (!defined && std::find(sources.begin(), sources.end(), mask) != sources.end())
                    uses[b] = true;
                const auto written = destination_registers(*instruction);
                if (std::find(written.begin(), written.end(), mask) != written.end()) {
                    definitions[b] = true;
                    defined = true;
                }
            }
        }
        std::vector<bool> live_in(raw_blocks.size(), false);
        std::deque<std::size_t> pending;
        std::vector<bool> queued(raw_blocks.size(), true);
        for (std::size_t b = raw_blocks.size(); b-- > 0;) pending.push_back(b);
        while (!pending.empty()) {
            if (budget == 0) return std::nullopt;
            --budget;
            const auto b = pending.front();
            pending.pop_front();
            queued[b] = false;
            const bool live_out = std::any_of(
                raw_blocks[b].successors.begin(), raw_blocks[b].successors.end(),
                [&](std::size_t successor) { return live_in[successor]; });
            const bool next = uses[b] || (live_out && !definitions[b]);
            if (next == live_in[b]) continue;
            live_in[b] = next;
            for (const auto predecessor : raw_blocks[b].predecessors) {
                if (!queued[predecessor]) {
                    pending.push_back(predecessor);
                    queued[predecessor] = true;
                }
            }
        }
        return live_in;
    }

    // This deliberately is not a general purity classifier. These are the
    // exact register-only operations emitted by the observed comparison; every
    // memory access, call, trap, synchronization, and unknown opcode is a root.
    static bool removable_mask_operation(const Instruction& instruction) {
        if (!instruction.predicate.empty()) return false;
        if (destination_registers(instruction).size() != 1) return false;
        if (instruction.opcode == "mov.pred" || instruction.opcode == "cvt.u16.u32")
            return instruction.operands.size() == 2;
        if (instruction.opcode == "and.pred" || instruction.opcode == "setp.eq.b16" ||
            instruction.opcode == "setp.eq.u32" || instruction.opcode == "and.b16")
            return instruction.operands.size() == 3;
        return instruction.opcode == "prmt.b32" && instruction.operands.size() == 4;
    }

    void remove_dead_mask_operations(const std::unordered_set<std::size_t>& specialized) {
        using Registers = std::unordered_set<std::string>;
        Registers tracked;
        for (const auto b : specialized)
            for (const auto* instruction : raw_blocks[b].instructions)
                if (removable_mask_operation(*instruction))
                    for (const auto& reg : destination_registers(*instruction)) tracked.insert(reg);
        if (tracked.empty()) return;
        struct Access { Registers reads, writes; bool removable = false; };
        std::vector<std::vector<Access>> accesses(raw_blocks.size());
        for (std::size_t b = 0; b < raw_blocks.size(); ++b) {
            for (const auto* instruction : raw_blocks[b].instructions) {
                Access access;
                access.removable = specialized.contains(b) &&
                                   removable_mask_operation(*instruction);
                for (const auto& reg : source_registers(*instruction))
                    if (tracked.contains(reg)) access.reads.insert(reg);
                for (const auto& reg : destination_registers(*instruction))
                    if (tracked.contains(reg)) access.writes.insert(reg);
                accesses[b].push_back(std::move(access));
            }
        }
        std::vector<Registers> live_in(raw_blocks.size());
        std::deque<std::size_t> pending;
        std::vector<bool> queued(raw_blocks.size(), true);
        for (std::size_t b = raw_blocks.size(); b-- > 0;) pending.push_back(b);
        std::size_t budget = 16777216;
        while (!pending.empty()) {
            if (budget == 0) return;
            --budget;
            const auto b = pending.front();
            pending.pop_front();
            queued[b] = false;
            Registers live;
            for (const auto successor : raw_blocks[b].successors) {
                if (live_in[successor].size() > budget) return;
                budget -= live_in[successor].size();
                live.insert(live_in[successor].begin(), live_in[successor].end());
            }
            for (std::size_t i = accesses[b].size(); i-- > 0;) {
                const auto& access = accesses[b][i];
                const bool dead = access.removable &&
                    std::none_of(access.writes.begin(), access.writes.end(),
                                 [&](const auto& reg) { return live.contains(reg); });
                if (dead) continue;
                for (const auto& reg : access.writes) live.erase(reg);
                live.insert(access.reads.begin(), access.reads.end());
            }
            if (live == live_in[b]) continue;
            live_in[b] = std::move(live);
            for (const auto predecessor : raw_blocks[b].predecessors) {
                if (!queued[predecessor]) {
                    pending.push_back(predecessor);
                    queued[predecessor] = true;
                }
            }
        }
        for (const auto b : specialized) {
            Registers live;
            for (const auto successor : raw_blocks[b].successors)
                live.insert(live_in[successor].begin(), live_in[successor].end());
            auto& instructions = raw_blocks[b].instructions;
            for (std::size_t i = instructions.size(); i-- > 0;) {
                const auto& access = accesses[b][i];
                if (access.removable &&
                    std::none_of(access.writes.begin(), access.writes.end(),
                                 [&](const auto& reg) { return live.contains(reg); })) {
                    instructions[i] = nullptr;
                    continue;
                }
                for (const auto& reg : access.writes) live.erase(reg);
                live.insert(access.reads.begin(), access.reads.end());
            }
            std::erase(instructions, nullptr);
        }
    }

    // Split only blocks reached with both literal mask states. The existing
    // blocks become the true version; copied instructions form the false
    // version, where AND absorption can expose dead pure comparison work.
    bool specialize_mask(const std::string& mask) {
        if (raw_blocks.empty()) return false;
        const auto liveness = mask_liveness(mask);
        if (!liveness) return false;
        const std::size_t original_count = raw_blocks.size();
        std::vector<unsigned char> states(original_count, 0);
        std::deque<std::pair<std::size_t, MaskState>> pending;
        const auto enqueue = [&](std::size_t block, MaskState state) {
            const auto bit = static_cast<unsigned char>(state_bit(state));
            if (states[block] & bit) return;
            states[block] |= bit;
            pending.emplace_back(block, state);
        };
        enqueue(0, MaskState::Unknown);
        std::size_t budget = 16777216;
        while (!pending.empty()) {
            const auto [block, state] = pending.front();
            pending.pop_front();
            if (raw_blocks[block].instructions.size() > budget) return false;
            budget -= raw_blocks[block].instructions.size();
            for (std::size_t edge = 0; edge < raw_blocks[block].successors.size(); ++edge) {
                const auto outgoing = transfer_mask_edge(raw_blocks[block], mask, state, edge);
                if (outgoing) enqueue(raw_blocks[block].successors[edge], *outgoing);
            }
        }

        std::vector<bool> mixed(original_count, false);
        std::size_t clone_instructions = 0;
        for (std::size_t b = 0; b < original_count; ++b) {
            if (!(*liveness)[b]) continue;
            const bool unknown = states[b] & state_bit(MaskState::Unknown);
            const bool absent = states[b] & state_bit(MaskState::False);
            const bool present = states[b] & state_bit(MaskState::True);
            if (unknown && (absent || present)) {
                return false;
            }
            if (!absent || !present) continue;
            mixed[b] = true;
            clone_instructions += raw_blocks[b].instructions.size();
        }
        const auto mixed_count = std::count(mixed.begin(), mixed.end(), true);
        if (mixed_count == 0 || cloned_blocks + mixed_count > 4096 ||
            clone_instructions > 131072 - cloned_instructions) return false;

        std::vector<std::vector<std::size_t>> original_successors;
        original_successors.reserve(original_count);
        for (std::size_t b = 0; b < original_count; ++b)
            original_successors.push_back(raw_blocks[b].successors);

        struct EdgePlan { bool reachable = false; bool false_version = false; };
        std::vector<std::vector<EdgePlan>> plans(original_count);
        for (std::size_t b = 0; b < original_count; ++b) {
            plans[b].resize(original_successors[b].size());
            const unsigned char inputs = mixed[b]
                ? static_cast<unsigned char>(state_bit(MaskState::True)) : states[b];
            for (std::size_t edge = 0; edge < original_successors[b].size(); ++edge) {
                std::optional<bool> false_version;
                for (const auto state : {MaskState::Unknown, MaskState::False, MaskState::True}) {
                    if (!(inputs & state_bit(state))) continue;
                    const auto outgoing = transfer_mask_edge(raw_blocks[b], mask, state, edge);
                    if (!outgoing) continue;
                    const auto target = original_successors[b][edge];
                    if (mixed[target] && *outgoing == MaskState::Unknown) {
                        return false;
                    }
                    const bool use_false = mixed[target] && *outgoing == MaskState::False;
                    if (false_version && *false_version != use_false) {
                        return false;
                    }
                    false_version = use_false;
                }
                if (false_version) plans[b][edge] = {true, *false_version};
            }
        }

        std::vector<std::size_t> false_clones(original_count, raw_blocks.size());
        std::unordered_set<std::size_t> specialized;
        for (std::size_t b = 0; b < original_count; ++b) {
            if (!mixed[b]) continue;
            RawBlock clone;
            clone.id = builder.next_block();
            clone.name = raw_blocks[b].name + "_mask_false_" + std::to_string(raw_blocks.size());
            for (const auto* instruction : raw_blocks[b].instructions) {
                storage.push_back(*instruction);
                record_instruction_origin(origins, &storage.back(), instruction);
                clone.instructions.push_back(&storage.back());
            }
            clone.successors = raw_blocks[b].successors;
            false_clones[b] = raw_blocks.size();
            specialized.insert(raw_blocks.size());
            raw_blocks.push_back(std::move(clone));
        }
        cloned_blocks += mixed_count;
        cloned_instructions += clone_instructions;

        const auto mapped = [&](std::size_t target, bool use_false) {
            return use_false ? false_clones[target] : target;
        };
        for (std::size_t b = 0; b < original_count; ++b) {
            if (states[b] == 0) continue;
            std::vector<std::size_t> successors;
            for (std::size_t edge = 0; edge < original_successors[b].size(); ++edge)
                if (plans[b][edge].reachable)
                    successors.push_back(mapped(original_successors[b][edge],
                                                plans[b][edge].false_version));
            raw_blocks[b].successors = std::move(successors);
        }
        for (std::size_t b = 0; b < original_count; ++b) {
            if (!mixed[b]) continue;
            auto& clone = raw_blocks[false_clones[b]];
            auto source = raw_blocks[b];
            source.successors = original_successors[b];
            std::vector<std::size_t> successors;
            for (std::size_t edge = 0; edge < original_successors[b].size(); ++edge) {
                const auto outgoing = transfer_mask_edge(source, mask,
                                                         MaskState::False, edge);
                if (!outgoing) continue;
                const auto target = original_successors[b][edge];
                successors.push_back(mapped(target, mixed[target] &&
                                                   *outgoing == MaskState::False));
            }
            clone.successors = std::move(successors);
        }

        for (const auto index : specialized) {
            auto& block = raw_blocks[index];
            std::map<std::string, bool> known{{mask, false}};
            for (auto*& instruction : block.instructions) {
                const auto written = destination_registers(*instruction);
                const auto value = predicate_value(*instruction, known);
                for (const auto& reg : written) known.erase(reg);
                if (root_opcode(instruction->opcode) == "call")
                    invalidate_call(*instruction, known);
                if (!value || written.size() != 1) continue;
                Instruction replacement = *instruction;
                replacement.opcode = "mov.pred";
                replacement.operands = {written[0], *value ? "1" : "0"};
                replacement.predicate.clear();
                storage.push_back(std::move(replacement));
                record_instruction_origin(origins, &storage.back(), instruction);
                instruction = &storage.back();
                known[written[0]] = *value;
            }
            if (!block.successors.empty() && !block.instructions.empty()) {
                const auto* tail = block.instructions.back();
                if (root_opcode(tail->opcode) == "bra" && !tail->predicate.empty()) {
                    const auto [predicate, inverted] = normalized_predicate(tail->predicate);
                    const auto outcome = known.find(predicate);
                    if (outcome != known.end()) {
                        const auto successor = block.successors.size() == 1
                            ? block.successors[0]
                            : block.successors[(outcome->second != inverted) ? 0 : 1];
                        Instruction branch = *tail;
                        branch.predicate.clear();
                        branch.operands = {raw_blocks[successor].name};
                        storage.push_back(std::move(branch));
                        record_instruction_origin(origins, &storage.back(), tail);
                        block.instructions.back() = &storage.back();
                        block.successors = {successor};
                    }
                }
            }
        }
        for (auto& block : raw_blocks) block.predecessors.clear();
        for (std::size_t b = 0; b < raw_blocks.size(); ++b)
            for (const auto successor : raw_blocks[b].successors)
                raw_blocks[successor].predecessors.push_back(b);
        remove_dead_mask_operations(specialized);
        return true;
    }

    // Limit candidates to predicates assigned both literals and consumed by an
    // AND chain. A bounded transitive score prioritizes the large generated
    // comparisons while the shared clone budgets limit cumulative growth.
    void specialize_masked_predicates() {
        std::map<std::string, unsigned> literal_writes;
        std::unordered_map<std::string, std::vector<std::string>> absorbed_by;
        for (const auto& block : raw_blocks) {
            for (const auto* instruction : block.instructions) {
                const auto written = destination_registers(*instruction);
                if (instruction->opcode == "mov.pred" && instruction->predicate.empty() &&
                    instruction->operands.size() == 2 && written.size() == 1) {
                    const auto value = trim(instruction->operands[1]);
                    if (value == "0") literal_writes[written[0]] |= 1;
                    if (value == "1" || value == "-1") literal_writes[written[0]] |= 2;
                }
                if (instruction->opcode != "and.pred" || !instruction->predicate.empty() ||
                    instruction->operands.size() != 3 || written.size() != 1) continue;
                for (const std::size_t operand : {1u, 2u}) {
                    const auto source = trim(instruction->operands[operand]);
                    if (first_register(source) == source)
                        absorbed_by[source].push_back(written[0]);
                }
            }
        }
        std::vector<std::pair<std::size_t, std::string>> candidates;
        for (const auto& [mask, writes] : literal_writes) {
            if (writes != 3 || !absorbed_by.contains(mask)) continue;
            std::unordered_set<std::string> closure{mask};
            std::deque<std::string> pending{mask};
            while (!pending.empty()) {
                const auto current = pending.front();
                pending.pop_front();
                const auto found = absorbed_by.find(current);
                if (found == absorbed_by.end()) continue;
                for (const auto& next : found->second)
                    if (closure.insert(next).second) pending.push_back(next);
                if (closure.size() > 1024) break;
            }
            if (closure.size() > 1024) continue;
            candidates.emplace_back(closure.size(), mask);
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });
        for (const auto& [score, mask] : candidates) {
            (void)score;
            specialize_mask(mask);
        }
    }

    // Duplicate only a bounded chain of successors whose branch is implied by
    // the incoming edge. Retain every non-branch instruction, including guarded
    // loads: this exposes infeasible paths to SSA without inventing definitions.
    void thread_predicate_edges() {
        const auto original_count = raw_blocks.size();
        const auto entry_constants = constant_entries();
        for (std::size_t index = 0; index < original_count; ++index) {
            const RawBlock source = raw_blocks[index];
            if (source.instructions.empty() || source.successors.empty() ||
                source.successors.size() > 2) continue;
            const auto* branch = source.instructions.back();
            const bool conditional = root_opcode(branch->opcode) == "bra" &&
                                     !branch->predicate.empty();
            const auto [pred, inverted] = normalized_predicate(branch->predicate);
            std::optional<ComparisonFact> comparison;
            auto expressions = expressions_before(index);
            bool combined = false;
            auto constants = entry_constants[index];
            for (std::size_t j = 0; j < source.instructions.size(); ++j) {
                const auto& instruction = *source.instructions[j];
                const auto written = destination_registers(instruction);
                const auto value = predicate_value(instruction, constants);
                for (const auto& reg : written) constants.erase(reg);
                if (root_opcode(instruction.opcode) == "call") invalidate_call(instruction, constants);
                if (value && written.size() == 1) constants[written[0]] = *value;
                for (const auto& reg : written)
                    if (comparison && (reg == pred || writes_fact(reg, *comparison)))
                        comparison.reset();
                if (root_opcode(instruction.opcode) == "call") comparison.reset();
                if (std::find(written.begin(), written.end(), pred) != written.end()) combined = false;
                expressions.observe(instruction);
                if (written.size() == 1 && written[0] == pred) {
                    comparison = expressions.comparison(instruction);
                    combined = instruction.opcode == "or.pred" && instruction.predicate.empty();
                }
            }
            // Taking an edge establishes its predicate independently of the
            // comparison that produced it. Operand facts still require their
            // separate proof; a repeated unchanged predicate does not.
            if (!comparison && !combined && constants.empty() && !conditional) continue;
            for (std::size_t edge = 0; edge < source.successors.size(); ++edge) {
                auto known = constants;
                if (conditional) known[pred] = (edge == 0) != inverted;
                auto fact = comparison;
                auto edge_expressions = expressions;
                if (fact) fact->positive = known[pred] == fact->positive;
                auto parent = index;
                auto parent_edge = edge;
                auto current = source.successors[edge];
                std::unordered_set<std::size_t> visited{index};
                std::vector<RawBlock> prefix;
                std::size_t inspected = 0;
                while (visited.insert(current).second) {
                    const RawBlock target = raw_blocks[current];
                    if (target.instructions.empty() || target.instructions.size() > 256 - inspected ||
                        target.successors.empty() || target.successors.size() > 2) break;
                    inspected += target.instructions.size();
                    const auto* tail = target.instructions.back();
                    const bool conditional_tail = target.successors.size() == 2 &&
                        root_opcode(tail->opcode) == "bra" && !tail->predicate.empty();
                    if (target.successors.size() == 2 && !conditional_tail) break;
                    const auto body_size = target.instructions.size() - (conditional_tail ? 1 : 0);
                    for (std::size_t j = 0; j < body_size; ++j) {
                        const auto& instruction = *target.instructions[j];
                        const auto written = destination_registers(instruction);
                        auto value = predicate_value(instruction, known);
                        if (instruction.predicate.empty()) {
                            const auto next = edge_expressions.comparison(instruction);
                            if (next && fact && next->left == fact->left && next->right == fact->right &&
                                next->type == fact->type && next->relation == fact->relation)
                                value = next->positive == fact->positive;
                        }
                        for (const auto& reg : written) {
                            known.erase(reg);
                            if (fact && writes_fact(reg, *fact)) fact.reset();
                        }
                        if (root_opcode(instruction.opcode) == "call") { invalidate_call(instruction, known); fact.reset(); }
                        edge_expressions.observe(instruction);
                        if (value && written.size() == 1) known[written[0]] = *value;
                    }
                    if (target.successors.size() == 1) {
                        prefix.push_back(target);
                        current = target.successors[0];
                        continue;
                    }
                    const auto [tail_pred, tail_inverted] = normalized_predicate(tail->predicate);
                    const auto outcome = known.find(tail_pred);
                    if (outcome == known.end()) break;
                    const auto successor = target.successors[(outcome->second != tail_inverted) ? 0 : 1];
                    // Do not clone a straight-line prefix until it leads to a
                    // proven branch. Preserve every operation along that path.
                    bool exhausted = false;
                    for (const auto& straight : prefix) {
                        const auto cloned = clone_edge(parent, parent_edge, straight, straight.successors[0]);
                        if (!cloned) { exhausted = true; break; }
                        parent = *cloned;
                        parent_edge = 0;
                    }
                    if (exhausted) break;
                    prefix.clear();
                    const auto cloned = clone_edge(parent, parent_edge, target, successor);
                    if (!cloned) break;
                    parent = *cloned;
                    parent_edge = 0;
                    current = successor;
                }
            }
        }
    }

    // Seed the select proof from a bounded local history of integer comparisons.
    // Equal inputs and types establish predicate equality or complementation;
    // writes and calls discard facts before any rewrite is considered.
    static std::map<std::string, bool> comparison_aliases_before(
        const RawBlock& block, std::size_t index, const std::string& selected) {
        static const std::regex comparison(R"(^setp\.(eq|ne)\.([bu](16|32|64))$)");
        static const std::regex integer(R"(^-?[0-9]+$)");
        std::map<std::string, ComparisonFact> facts;
        for (std::size_t i = index > 64 ? index - 64 : 0; i < index; ++i) {
            const auto& instruction = *block.instructions[i];
            for (const auto& written : destination_registers(instruction)) {
                std::erase_if(facts, [&](const auto& item) {
                    return item.first == written || item.second.left == written || item.second.right == written;
                });
            }
            if (root_opcode(instruction.opcode) == "call") facts.clear();
            std::smatch match;
            if (!instruction.predicate.empty() || instruction.operands.size() != 3 ||
                !std::regex_match(instruction.opcode, match, comparison)) continue;
            const auto predicate = trim(instruction.operands[0]);
            if (predicate.empty() || first_register(predicate) != predicate) continue;
            auto left = trim(instruction.operands[1]);
            auto right = trim(instruction.operands[2]);
            const auto sources = source_registers(instruction);
            const auto scalar = [&](const std::string& operand) {
                return std::regex_match(operand, integer) ||
                    (!operand.empty() && first_register(operand) == operand &&
                     std::find(sources.begin(), sources.end(), operand) != sources.end() &&
                     !operand.starts_with("%globaltimer") && !operand.starts_with("%pm"));
            };
            if (!scalar(left) || !scalar(right)) continue;
            if (right < left) std::swap(left, right);
            facts[predicate] = ComparisonFact{left, right, match[2], "eq", match[1] == "eq"};
        }
        std::map<std::string, bool> aliases{{selected, true}};
        const auto found = facts.find(selected);
        if (found == facts.end()) return aliases;
        for (const auto& [predicate, fact] : facts) {
            if (fact.left == found->second.left && fact.right == found->second.right && fact.type == found->second.type)
                aliases[predicate] = fact.positive == found->second.positive;
        }
        return aliases;
    }

    void remove_unobserved_self_selects() {
        for (RawBlock& block : raw_blocks) {
            if (block.instructions.empty() || block.successors.size() != 2) continue;
            const Instruction* branch = block.instructions.back();
            if (root_opcode(branch->opcode) != "bra" || branch->predicate.empty()) continue;
            const auto [predicate, inverted] = normalized_predicate(branch->predicate);
            for (std::size_t index = 0; index + 1 < block.instructions.size(); ++index) {
                const Instruction* select = block.instructions[index];
                if ((select->opcode != "selp.b32" && select->opcode != "selp.b64") ||
                    !select->predicate.empty() || select->operands.size() != 4 ||
                    first_register(select->operands[3]) != trim(select->operands[3])) continue;
                const std::string destination = trim(select->operands[0]);
                const std::string source = trim(select->operands[1]);
                const int width = select->opcode == "selp.b32" ? 32 : 64;
                // Do not turn malformed select operands into valid mov tuples.
                // Keep this proof restricted to matching scalar registers.
                if (first_register(source) != source ||
                    ptx_register_container_bits(source) != width ||
                    ptx_register_container_bits(destination) != width) continue;
                if (first_register(destination) != destination ||
                    trim(select->operands[2]) != destination ||
                    trim(select->operands[1]) == destination) continue;
                const std::string select_predicate = trim(select->operands[3]);
                auto predicate_aliases = comparison_aliases_before(block, index, select_predicate);
                bool safe = true;
                for (std::size_t i = index + 1; i + 1 < block.instructions.size(); ++i) {
                    const auto sources = source_registers(*block.instructions[i]);
                    const auto destinations = destination_registers(*block.instructions[i]);
                    if (std::find(sources.begin(), sources.end(), destination) != sources.end() ||
                        std::find(destinations.begin(), destinations.end(), destination) != destinations.end() ||
                        std::find(destinations.begin(), destinations.end(), select_predicate) != destinations.end()) safe = false;
                    const Instruction& middle = *block.instructions[i];
                    std::optional<bool> alias;
                    if (middle.predicate.empty() && middle.operands.size() == 2 &&
                        (middle.opcode == "not.pred" || middle.opcode == "mov.pred")) {
                        auto found = predicate_aliases.find(trim(middle.operands[1]));
                        if (found != predicate_aliases.end())
                            alias = found->second != (middle.opcode == "not.pred");
                    }
                    for (const auto& written : destinations) predicate_aliases.erase(written);
                    if (alias && destinations.size() == 1) predicate_aliases[destinations[0]] = *alias;
                }
                if (!safe) continue;
                const auto branch_alias = predicate_aliases.find(predicate);
                if (branch_alias == predicate_aliases.end()) continue;
                const bool branch_on_selected = branch_alias->second != inverted;
                const std::size_t false_edge = block.successors[branch_on_selected ? 1 : 0];
                // Follow the false edge until an unconditional overwrite or
                // this same select. Cycles are safe only if no path reads the
                // old value. A subsequent true edge supplies the new value.
                std::vector<std::size_t> pending{false_edge};
                std::unordered_set<std::size_t> visited;
                while (!pending.empty() && safe) {
                    const auto current = pending.back();
                    pending.pop_back();
                    if (!visited.insert(current).second) continue;
                    bool killed = false;
                    for (const Instruction* instruction : raw_blocks[current].instructions) {
                        if (instruction == select) { killed = true; break; }
                        const auto sources = source_registers(*instruction);
                        if (std::find(sources.begin(), sources.end(), destination) != sources.end()) {
                            safe = false;
                            break;
                        }
                        const auto destinations = destination_registers(*instruction);
                        if (instruction->predicate.empty() &&
                            std::find(destinations.begin(), destinations.end(), destination) != destinations.end()) {
                            killed = true;
                            break;
                        }
                    }
                    if (!killed) {
                        pending.insert(pending.end(), raw_blocks[current].successors.begin(),
                                       raw_blocks[current].successors.end());
                    }
                }
                if (!safe) continue;

                Instruction replacement = *select;
                replacement.opcode = select->opcode == "selp.b32" ? "mov.b32" : "mov.b64";
                replacement.operands = {select->operands[0], select->operands[1]};
                storage.push_back(std::move(replacement));
                record_instruction_origin(origins, &storage.back(), select);
                block.instructions[index] = &storage.back();
            }
        }
    }

    // Ordinary register liveness treats every mov source as a use, even when
    // its destination only feeds another discarded copy. Compute demand from
    // non-copy instructions instead so an unobserved loop-carried copy cycle
    // does not require an invented initial SSA value.
    void remove_unobserved_copies() {
        struct Access {
            std::vector<std::string> reads;
            std::vector<std::string> writes;
            bool copy = false;
        };
        std::vector<std::vector<Access>> accesses(raw_blocks.size());
        std::unordered_set<std::string> copy_destinations;
        for (std::size_t b = 0; b < raw_blocks.size(); ++b) {
            for (const auto* instruction : raw_blocks[b].instructions) {
                Access access{source_registers(*instruction), destination_registers(*instruction)};
                const auto root = root_opcode(instruction->opcode);
                // Reduction addresses are inputs, not register definitions.
                if (root == "red") {
                    access.writes.clear();
                    for (const auto& operand : instruction->operands) {
                        const auto regs = registers_in(operand);
                        access.reads.insert(access.reads.end(), regs.begin(), regs.end());
                    }
                }
                if (instruction->predicate.empty() && instruction->operands.size() == 2 &&
                    access.writes.size() == 1 && access.reads.size() == 1) {
                    const auto width = ptx_register_container_bits(access.writes[0]);
                    access.copy = (width == 16 || width == 32 || width == 64) &&
                        instruction->opcode == "mov.b" + std::to_string(width) &&
                        trim(instruction->operands[0]) == access.writes[0] &&
                        trim(instruction->operands[1]) == access.reads[0] &&
                        ptx_register_container_bits(access.reads[0]) == width;
                }
                if (!instruction->predicate.empty()) access.writes.clear();
                if (access.copy) copy_destinations.insert(access.writes[0]);
                accesses[b].push_back(std::move(access));
            }
        }
        if (copy_destinations.empty()) return;
        // Only registers defined by a removable copy affect a deletion decision.
        // Other instructions remain roots even when their results are unused;
        // their reads still demand every relevant copy operand.
        for (auto& block : accesses) {
            for (auto& access : block) {
                std::erase_if(access.reads, [&](const auto& reg) { return !copy_destinations.contains(reg); });
                std::erase_if(access.writes, [&](const auto& reg) { return !copy_destinations.contains(reg); });
            }
        }
        using Registers = std::unordered_set<std::string>;
        struct Summary {
            Registers roots;
            std::unordered_map<std::string, std::optional<std::string>> origins;
        };
        std::vector<Summary> summaries(raw_blocks.size());
        for (std::size_t b = 0; b < raw_blocks.size(); ++b) {
            auto& summary = summaries[b];
            const auto origin = [&](const std::string& reg) -> std::optional<std::string> {
                const auto found = summary.origins.find(reg);
                return found == summary.origins.end() ? std::optional<std::string>(reg) : found->second;
            };
            for (const auto& access : accesses[b]) {
                if (access.copy) {
                    summary.origins[access.writes[0]] = access.reads.empty() ? std::nullopt : origin(access.reads[0]);
                } else {
                    for (const auto& reg : access.reads)
                        if (const auto input = origin(reg)) summary.roots.insert(*input);
                    for (const auto& reg : access.writes) summary.origins[reg] = std::nullopt;
                }
            }
        }
        std::vector<Registers> live_in(raw_blocks.size());
        auto transfer = [](Registers& live, const Access& access) {
            if (access.copy && !live.contains(access.writes[0])) return;
            for (const auto& reg : access.writes) live.erase(reg);
            live.insert(access.reads.begin(), access.reads.end());
        };
        // Summarize each block once: a copy either forwards one block input or
        // a locally defined value. Iteration need not rescan large crypto bodies.
        // Bound work and storage independently; this analysis never clones code.
        std::size_t budget = 4194304;
        std::size_t stored_registers = 0;
        std::deque<std::size_t> pending;
        std::vector<bool> queued(raw_blocks.size(), true);
        for (std::size_t b = raw_blocks.size(); b-- > 0;) pending.push_back(b);
        while (!pending.empty()) {
            const auto b = pending.front();
            pending.pop_front();
            queued[b] = false;
            Registers live = summaries[b].roots;
            if (live.size() >= budget) return;
            budget -= live.size() + 1;
            for (const auto successor : raw_blocks[b].successors) {
                if (live_in[successor].size() > budget) return;
                budget -= live_in[successor].size();
                for (const auto& reg : live_in[successor]) {
                    const auto found = summaries[b].origins.find(reg);
                    if (found == summaries[b].origins.end()) live.insert(reg);
                    else if (found->second) live.insert(*found->second);
                }
            }
            if (live != live_in[b]) {
                stored_registers -= live_in[b].size();
                stored_registers += live.size();
                if (stored_registers > 1048576) return;
                live_in[b] = std::move(live);
                for (const auto predecessor : raw_blocks[b].predecessors) {
                    if (!queued[predecessor]) {
                        pending.push_back(predecessor);
                        queued[predecessor] = true;
                    }
                }
            }
        }
        for (std::size_t b = 0; b < raw_blocks.size(); ++b) {
            Registers live;
            for (const auto successor : raw_blocks[b].successors)
                live.insert(live_in[successor].begin(), live_in[successor].end());
            auto& instructions = raw_blocks[b].instructions;
            for (std::size_t i = accesses[b].size(); i-- > 0;) {
                const auto& access = accesses[b][i];
                if (access.copy && !live.contains(access.writes[0])) instructions[i] = nullptr;
                else transfer(live, access);
            }
            std::erase(instructions, nullptr);
        }
    }

};

}  // namespace

void remove_unreachable_blocks(std::vector<RawBlock>& blocks) {
    // Once every incoming edge was specialized, the original join can become
    // unreachable. Do not ask register SSA to invent definitions for that dead
    // copy. Remap only CFG indices; block IDs and instruction ownership survive.
    if (!blocks.empty()) {
        std::vector<bool> reachable(blocks.size(), false);
        std::vector<std::size_t> pending{0};
        while (!pending.empty()) {
            const auto index = pending.back();
            pending.pop_back();
            if (reachable[index]) continue;
            reachable[index] = true;
            pending.insert(pending.end(), blocks[index].successors.begin(),
                           blocks[index].successors.end());
        }
        std::vector<std::size_t> remap(blocks.size());
        std::vector<RawBlock> kept;
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            if (!reachable[i]) continue;
            remap[i] = kept.size();
            kept.push_back(std::move(blocks[i]));
        }
        for (auto& block : kept)
            for (auto& successor : block.successors) successor = remap[successor];
        blocks = std::move(kept);
    }
    for (auto& block : blocks) block.predecessors.clear();
    for (std::size_t index = 0; index < blocks.size(); ++index)
        for (auto successor : blocks[index].successors)
            blocks[successor].predecessors.push_back(index);
}

void simplify_guarded_paths(std::vector<RawBlock>& blocks, Builder& builder,
                            std::deque<Instruction>& storage,
                            const cumetal::ptx::EntryFunction* function,
                            InstructionOrigins* origins) {
    GuardedPaths paths{blocks, builder, storage, function, origins};
    paths.specialize_masked_predicates();
    for (auto& block : blocks) block.predecessors.clear();
    for (std::size_t index = 0; index < blocks.size(); ++index)
        for (auto successor : blocks[index].successors)
            blocks[successor].predecessors.push_back(index);
    paths.thread_threshold_edges();
    // Threshold specialization rewires edges and appends clones. The next
    // proof consumes predecessor topology, so refresh it before proceeding.
    for (auto& block : blocks) block.predecessors.clear();
    for (std::size_t index = 0; index < blocks.size(); ++index)
        for (auto successor : blocks[index].successors)
            blocks[successor].predecessors.push_back(index);
    paths.thread_predicate_edges();
    for (auto& block : blocks) block.predecessors.clear();
    for (std::size_t index = 0; index < blocks.size(); ++index)
        for (auto successor : blocks[index].successors)
            blocks[successor].predecessors.push_back(index);
    paths.remove_unobserved_self_selects();
    paths.remove_unobserved_copies();
}

}  // namespace cumetal::ir::detail
