#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>
#include <utility>

namespace cumetal::ir::detail {

// Immediate dominators in reverse postorder, followed by dominator-tree
// intervals for constant-time queries. Storage is linear in blocks and edges.
// A virtual root preserves the verifier's existing treatment of entry and
// predecessor-free blocks; closed unreachable components keep universal sets.
class Dominance {
    std::vector<std::size_t> begin_, end_;
    std::size_t missing_;
public:
    explicit Dominance(const std::vector<std::vector<std::size_t>>& predecessors) {
        const auto count = predecessors.size();
        const auto root = count;
        missing_ = count + 1;
        std::vector<std::vector<std::size_t>> successors(count + 1);
        for (std::size_t node = 0; node < count; ++node) {
            if (node == 0 || predecessors[node].empty()) successors[root].push_back(node);
            // Entry's dominator set is fixed, even if a back edge targets it.
            if (node != 0)
                for (const auto pred : predecessors[node]) successors[pred].push_back(node);
        }
        std::vector<bool> visited(count + 1, false);
        std::vector<std::pair<std::size_t, std::size_t>> stack{{root, 0}};
        std::vector<std::size_t> order;
        visited[root] = true;
        while (!stack.empty()) {
            auto& [node, next] = stack.back();
            if (next == successors[node].size()) {
                order.push_back(node);
                stack.pop_back();
            } else {
                const auto child = successors[node][next++];
                if (!visited[child]) {
                    visited[child] = true;
                    stack.emplace_back(child, 0);
                }
            }
        }
        std::reverse(order.begin(), order.end());
        std::vector<std::size_t> rank(count + 1, missing_), parent(count + 1, missing_);
        for (std::size_t i = 0; i < order.size(); ++i) rank[order[i]] = i;
        parent[root] = root;
        auto intersect = [&](std::size_t left, std::size_t right) {
            while (left != right) {
                while (rank[left] > rank[right]) left = parent[left];
                while (rank[right] > rank[left]) right = parent[right];
            }
            return left;
        };
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto node : order) {
                if (node == root) continue;
                auto next = missing_;
                if (node == 0 || predecessors[node].empty()) next = root;
                else for (const auto pred : predecessors[node]) {
                    if (parent[pred] == missing_) continue;
                    next = next == missing_ ? pred : intersect(next, pred);
                }
                if (parent[node] != next) {
                    parent[node] = next;
                    changed = true;
                }
            }
        }
        std::vector<std::vector<std::size_t>> children(count + 1);
        for (std::size_t node = 0; node < count; ++node)
            if (parent[node] != missing_) children[parent[node]].push_back(node);
        begin_.assign(count + 1, missing_);
        end_.assign(count + 1, missing_);
        stack = {{root, 0}};
        std::size_t clock = 0;
        begin_[root] = clock++;
        while (!stack.empty()) {
            auto& [node, next] = stack.back();
            if (next == children[node].size()) {
                end_[node] = clock;
                stack.pop_back();
            } else {
                const auto child = children[node][next++];
                begin_[child] = clock++;
                stack.emplace_back(child, 0);
            }
        }
    }
    bool dominates(std::size_t definition, std::size_t use) const {
        // Preserve the old verifier's fixed-point convention for closed,
        // unreachable components: their dominator set remains universal.
        return begin_[use] == missing_ ||
               (begin_[definition] != missing_ && begin_[definition] <= begin_[use] &&
                begin_[use] < end_[definition]);
    }
};

}  // namespace cumetal::ir::detail
