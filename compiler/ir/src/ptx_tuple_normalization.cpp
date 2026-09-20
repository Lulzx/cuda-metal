#include "ptx_tuple_normalization.h"
#include "ptx_text.h"

#include <algorithm>
#include <charconv>
#include <map>

namespace cumetal::ir::detail {
namespace {

struct Budget {
    TupleNormalizationLimits remaining;
    bool exhausted = false;

    bool spend(std::size_t& allowance, std::size_t count = 1) {
        if (count > allowance) {
            exhausted = true;
            return false;
        }
        allowance -= count;
        return true;
    }
    bool text(std::string_view value) { return spend(remaining.text_bytes, value.size()); }
};

struct Width {
    unsigned bits = 0;
    bool integer = false;
};

Width declared_width(std::string_view type) {
    if (type == "b16" || type == "u16" || type == "s16")
        return {16, true};
    if (type == "b32" || type == "u32" || type == "s32")
        return {32, true};
    if (type == "b64" || type == "u64" || type == "s64")
        return {64, true};
    // Existing bitwise pack/extract moves also permit floating bit containers.
    if (type == "f16")
        return {16, false};
    if (type == "f32")
        return {32, false};
    if (type == "f64")
        return {64, false};
    return {};
}

class Declarations {
  public:
    Declarations(const cumetal::ptx::EntryFunction& function, Budget& budget)
        : function_(function), budget_(budget) {}

    Width lookup(const std::string& name) {
        if (!budget_.text(name))
            return {};
        if (const auto found = cache_.find(name); found != cache_.end())
            return found->second;
        if (cache_.size() >= budget_.remaining.declaration_entries) {
            budget_.exhausted = true;
            return {};
        }
        if (!indexed_) {
            indexed_ = true;
            if (function_.register_declarations.size() > budget_.remaining.declaration_entries) {
                budget_.exhausted = true;
                return {};
            }
            for (const auto& declaration : function_.register_declarations) {
                if (!budget_.spend(budget_.remaining.declaration_checks) || !budget_.text(declaration.name) ||
                    !budget_.text(declaration.type))
                    return {};
                const auto [found, inserted] =
                    exact_.emplace(declaration.name,
                                   declaration.function_scope ? declared_width(declaration.type) : Width{});
                if (!inserted)
                    found->second = {};
            }
        }
        const auto exact = exact_.find(name);
        bool matched = exact != exact_.end();
        Width width = matched ? exact->second : Width{};
        // Do not expand ranges, even if their count is near size_t's maximum.
        for (const auto& range : function_.register_ranges) {
            if (!budget_.spend(budget_.remaining.declaration_checks) || !budget_.text(range.prefix) ||
                !budget_.text(name))
                return {};
            if (!name.starts_with(range.prefix))
                continue;
            const auto suffix = std::string_view(name).substr(range.prefix.size());
            if (suffix.empty() || (suffix.size() > 1 && suffix.front() == '0'))
                continue;
            std::size_t index = 0;
            const auto parsed = std::from_chars(suffix.data(), suffix.data() + suffix.size(), index);
            if (parsed.ec != std::errc{} || parsed.ptr != suffix.data() + suffix.size() ||
                index >= range.count)
                continue;
            width = !matched && range.function_scope ? declared_width(range.type) : Width{};
            matched = true;
        }
        cache_.emplace(name, width);
        return width;
    }

  private:
    const cumetal::ptx::EntryFunction& function_;
    Budget& budget_;
    bool indexed_ = false;
    std::unordered_map<std::string, Width> exact_, cache_;
};

bool plain_register(const std::string& operand) {
    return !operand.empty() && first_register(operand) == operand;
}

std::vector<std::string> tuple(const std::string& operand) {
    const auto text = trim(operand);
    if (text.size() < 5 || text.front() != '{' || text.back() != '}')
        return {};
    const auto comma = text.find(',');
    if (comma == std::string::npos || text.find(',', comma + 1) != std::string::npos)
        return {};
    return {trim(text.substr(1, comma - 1)), trim(text.substr(comma + 1, text.size() - comma - 2))};
}

std::vector<std::string> writes(const Instruction& instruction) {
    if (root_opcode(instruction.opcode) == "call" && instruction.operands.size() == 3)
        return registers_in(instruction.operands.front());
    return destination_registers(instruction);
}

std::vector<std::string> reads(const Instruction& instruction) {
    // The general SSA helper filters builtin-looking names. Here declarations
    // decide register identity: even a declared %clock_payload is an observed
    // source and must not disappear from the whole-function use proof.
    std::vector<std::string> result;
    const std::size_t first = destination_registers(instruction).empty() ? 0 : 1;
    for (std::size_t i = first; i < instruction.operands.size(); ++i) {
        const auto names = registers_in(instruction.operands[i]);
        result.insert(result.end(), names.begin(), names.end());
    }
    if (!instruction.predicate.empty()) {
        const auto predicate = first_register(instruction.predicate);
        if (!predicate.empty())
            result.push_back(predicate);
    }
    return result;
}

struct Position {
    std::size_t block = 0, index = 0;
};
struct Usage {
    unsigned definitions = 0, reads = 0;
    Position consumer;
    std::size_t generation = 0;
};
struct Candidate {
    const Instruction* pack;
    const Instruction* consumer;
    std::string source;
    std::size_t generation, call_generation;
    Instruction replacement;
};

} // namespace

void remove_discarded_pack_halves(std::vector<RawBlock>& blocks, std::deque<Instruction>& storage,
                                  const cumetal::ptx::EntryFunction* function, InstructionOrigins* origins,
                                  TupleNormalizationLimits limits) {
    if (!function)
        return;
    Budget budget{limits};
    Declarations declarations(*function, budget);
    std::unordered_map<std::string, Usage> usage;
    // Count all occurrences before any mutation. Saturation avoids arithmetic
    // overflow and prevents a later rewrite from manufacturing a single use.
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        if (!budget.spend(budget.remaining.instruction_visits))
            return;
        for (std::size_t i = 0; i < blocks[b].instructions.size(); ++i) {
            if (!budget.spend(budget.remaining.instruction_visits))
                return;
            const auto& instruction = *blocks[b].instructions[i];
            if (!budget.text(instruction.opcode) || !budget.text(instruction.predicate))
                return;
            for (const auto& operand : instruction.operands)
                if (!budget.text(operand))
                    return;
            const auto count = [&](const std::string& reg, bool definition) {
                if (!budget.spend(budget.remaining.register_occurrences))
                    return false;
                if (!usage.contains(reg) && usage.size() >= limits.tracked_registers)
                    return false;
                auto& item = usage[reg];
                auto& occurrences = definition ? item.definitions : item.reads;
                occurrences = std::min(2u, occurrences + 1);
                if (!definition)
                    item.consumer = {b, i};
                return true;
            };
            for (const auto& reg : writes(instruction))
                if (!count(reg, true))
                    return;
            for (const auto& reg : reads(instruction))
                if (!count(reg, false))
                    return;
        }
    }

    std::vector<Candidate> replacements;
    std::size_t call_generation = 0;
    const auto width = [&](const std::string& reg) {
        return plain_register(reg) ? declarations.lookup(reg) : Width{};
    };
    const auto unread = [&](const std::string& reg) {
        const auto found = usage.find(reg);
        return found == usage.end() || found->second.reads == 0;
    };
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        if (!budget.spend(budget.remaining.instruction_visits))
            return;
        std::map<std::size_t, Candidate> pending;
        for (std::size_t i = 0; i < blocks[b].instructions.size(); ++i) {
            if (!budget.spend(budget.remaining.instruction_visits))
                return;
            const auto* instruction = blocks[b].instructions[i];
            // Read before writes: the consumer may overwrite its selected input.
            if (const auto found = pending.find(i); found != pending.end()) {
                const auto& candidate = found->second;
                if (usage.at(candidate.source).generation == candidate.generation &&
                    call_generation == candidate.call_generation)
                    replacements.push_back(std::move(found->second));
                pending.erase(found);
            }
            if (instruction->opcode == "mov.b64" && instruction->predicate.empty() &&
                instruction->operands.size() == 2) {
                const auto packed = trim(instruction->operands[0]);
                const auto halves = tuple(instruction->operands[1]);
                const auto found = usage.find(packed);
                if (plain_register(packed) && halves.size() == 2 && found != usage.end() &&
                    found->second.definitions == 1 && found->second.reads == 1 &&
                    found->second.consumer.block == b && found->second.consumer.index > i) {
                    const auto packed_width = width(packed);
                    const auto low_width = width(halves[0]), high_width = width(halves[1]);
                    const auto consumer_index = found->second.consumer.index;
                    const auto* consumer = blocks[b].instructions[consumer_index];
                    if (packed_width.bits == 64 && low_width.bits == 32 && high_width.bits == 32 &&
                        consumer->predicate.empty() && consumer->operands.size() == 2 &&
                        trim(consumer->operands[1]) == packed) {
                        int selected = -1;
                        Instruction replacement = *consumer;
                        if (consumer->opcode == "mov.b64") {
                            const auto lanes = tuple(consumer->operands[0]);
                            if (lanes.size() == 2) {
                                const auto discarded = [&](const std::string& lane) {
                                    return lane == "_" || (width(lane).bits == 32 && unread(lane));
                                };
                                const bool discard_low = discarded(lanes[0]);
                                const bool discard_high = discarded(lanes[1]);
                                if (discard_low != discard_high) {
                                    selected = discard_low ? 1 : 0;
                                    if (width(lanes[selected]).bits != 32)
                                        selected = -1;
                                    else {
                                        replacement.opcode = "mov.b32";
                                        replacement.operands = {lanes[selected], halves[selected]};
                                    }
                                }
                            }
                        } else if (consumer->opcode == "cvt.u16.u64" || consumer->opcode == "cvt.u32.u64") {
                            const auto destination = trim(consumer->operands[0]);
                            const auto destination_width = width(destination);
                            const unsigned format = consumer->opcode == "cvt.u16.u64" ? 16 : 32;
                            if (packed_width.integer && low_width.integer && high_width.integer &&
                                destination_width.integer && destination_width.bits >= format) {
                                selected = 0;
                                replacement.opcode = format == 16 ? "cvt.u16.u32" : "cvt.u32.u32";
                                replacement.operands = {consumer->operands[0], halves[0]};
                            }
                        }
                        if (selected >= 0) {
                            if (!budget.spend(budget.remaining.candidates))
                                return;
                            pending.emplace(consumer_index,
                                            Candidate{instruction, consumer, halves[selected],
                                                      usage.at(halves[selected]).generation, call_generation,
                                                      std::move(replacement)});
                        }
                    }
                }
            }
            if (budget.exhausted)
                return;
            for (const auto& reg : writes(*instruction)) {
                if (!budget.spend(budget.remaining.register_occurrences))
                    return;
                ++usage.at(reg).generation;
            }
            if (root_opcode(instruction->opcode) == "call")
                ++call_generation;
        }
    }
    if (replacements.empty())
        return;
    // Reserve the final scan before committing anything. Even a late exhausted
    // proof leaves all original demands for normal SSA validation.
    for (const auto& block : blocks)
        if (!budget.spend(budget.remaining.instruction_visits) ||
            !budget.spend(budget.remaining.instruction_visits, block.instructions.size()))
            return;

    std::unordered_map<const Instruction*, const Instruction*> actions;
    for (auto& replacement : replacements) {
        storage.push_back(std::move(replacement.replacement));
        record_instruction_origin(origins, &storage.back(), replacement.consumer);
        actions.emplace(replacement.consumer, &storage.back());
        actions.emplace(replacement.pack, nullptr);
    }
    for (auto& block : blocks) {
        std::size_t kept = 0;
        for (const auto* instruction : block.instructions) {
            if (const auto found = actions.find(instruction); found != actions.end())
                instruction = found->second;
            if (instruction)
                block.instructions[kept++] = instruction;
        }
        block.instructions.resize(kept);
    }
}

} // namespace cumetal::ir::detail
