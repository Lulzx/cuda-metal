#pragma once

#include <algorithm>
#include <functional>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Scalar/LoopUnrollPass.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Utils/LCSSA.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>
#include <map>
#include <vector>

namespace cumetal::ir {

// Unroll only tiny, explicitly hinted private-array loops. LLVM's general
// unroller otherwise also expands the much larger outer dot-product loops.
// Scalar replacement after each round exposes nested per-thread accumulators.
class PrivateArrayUnrollPass
    : public llvm::PassInfoMixin<PrivateArrayUnrollPass> {
  public:
    llvm::PreservedAnalyses run(llvm::Function &function,
                                llvm::FunctionAnalysisManager &analyses) {
        bool has_large_array = false;
        bool has_hint = false;
        for (auto &block : function) {
            for (auto &instruction : block) {
                if (auto *allocation =
                        llvm::dyn_cast<llvm::AllocaInst>(&instruction)) {
                    // Small arrays already scalarize well in Apple's compiler.
                    // Focus on the larger private arrays that can force indexed
                    // storage: at least 256 bytes (64 FP32 accumulators).
                    has_large_array |=
                        allocation->isStaticAlloca() &&
                        allocation->getAllocatedType()->isArrayTy() &&
                        function.getParent()->getDataLayout().getTypeAllocSize(
                            allocation->getAllocatedType()) >= 256;
                }
                if (auto *id =
                        instruction.getMetadata(llvm::LLVMContext::MD_loop)) {
                    has_hint |= requests_unroll(id);
                }
            }
        }
        if (!has_large_array || !has_hint)
            return llvm::PreservedAnalyses::all();

        bool changed = false;
        for (unsigned round = 0; round < 3; ++round) {
            llvm::FunctionPassManager canonicalize;
            canonicalize.addPass(llvm::LoopSimplifyPass{});
            canonicalize.addPass(llvm::LCSSAPass{});
            canonicalize.run(function, analyses);
            changed = true;
            auto &loops = analyses.getResult<llvm::LoopAnalysis>(function);
            auto &evolution =
                analyses.getResult<llvm::ScalarEvolutionAnalysis>(function);
            std::vector<llvm::Loop *> all_loops;
            std::vector<llvm::Loop *> selected;
            std::function<void(llvm::Loop *)> visit = [&](llvm::Loop *loop) {
                all_loops.push_back(loop);
                for (auto *child : loop->getSubLoops())
                    visit(child);
                if (!loop->getSubLoops().empty() ||
                    !requests_unroll(loop->getLoopID()))
                    return;
                // A pre-tested eight-iteration loop visits its header nine
                // times; SCEV reports header executions, not body executions.
                unsigned count = evolution.getSmallConstantTripCount(loop);
                if (count < 2 || count > 9)
                    return;
                unsigned instructions = 0;
                bool private_access = false;
                for (auto *block : loop->blocks()) {
                    for (auto &instruction : *block) {
                        if (++instructions > 1024 / count)
                            return;
                        // Never duplicate barriers, atomics, shuffles or calls
                        // whose CUDA SIMT semantics LLVM cannot model.
                        if (llvm::isa<llvm::CallBase>(instruction) ||
                            llvm::isa<llvm::AtomicRMWInst>(instruction) ||
                            llvm::isa<llvm::AtomicCmpXchgInst>(instruction) ||
                            llvm::isa<llvm::FenceInst>(instruction))
                            return;
                        const llvm::Value *pointer = nullptr;
                        if (auto *load =
                                llvm::dyn_cast<llvm::LoadInst>(&instruction)) {
                            if (load->isVolatile() || load->isAtomic())
                                return;
                            pointer = load->getPointerOperand();
                        } else if (auto *store =
                                       llvm::dyn_cast<llvm::StoreInst>(
                                           &instruction)) {
                            if (store->isVolatile() || store->isAtomic())
                                return;
                            pointer = store->getPointerOperand();
                        }
                        if (pointer && llvm::isa<llvm::AllocaInst>(
                                           llvm::getUnderlyingObject(pointer)))
                            private_access = true;
                    }
                }
                if (private_access)
                    selected.push_back(loop);
            };
            for (auto *loop : loops)
                visit(loop);
            if (selected.empty())
                break;

            // Suppress unrelated hints only while LLVM runs. Restore surviving
            // loop metadata afterwards, including explicit unroll-disable.
            std::map<llvm::MDNode *, llvm::MDNode *> originals;
            for (auto *loop : all_loops) {
                auto *old = loop->getLoopID();
                llvm::SmallVector<llvm::Metadata *, 8> operands{nullptr};
                if (old) {
                    for (unsigned i = 1; i < old->getNumOperands(); ++i) {
                        auto *node = llvm::dyn_cast_or_null<llvm::MDNode>(
                            old->getOperand(i));
                        auto *name =
                            node && node->getNumOperands()
                                ? llvm::dyn_cast_or_null<llvm::MDString>(
                                      node->getOperand(0))
                                : nullptr;
                        if (!name ||
                            !name->getString().starts_with("llvm.loop.unroll."))
                            operands.push_back(old->getOperand(i));
                    }
                }
                bool chosen = std::find(selected.begin(), selected.end(),
                                        loop) != selected.end();
                operands.push_back(llvm::MDNode::get(
                    function.getContext(),
                    llvm::MDString::get(function.getContext(),
                                        chosen ? "llvm.loop.unroll.full"
                                               : "llvm.loop.unroll.disable")));
                auto *id =
                    llvm::MDNode::getDistinct(function.getContext(), operands);
                id->replaceOperandWith(0, id);
                originals[id] = old;
                loop->setLoopID(id);
            }
            llvm::LoopUnrollOptions options(2, true);
            options.setPartial(false)
                .setRuntime(false)
                .setPeeling(false)
                .setUpperBound(false)
                .setFullUnrollMaxCount(9);
            llvm::FunctionPassManager optimize;
            optimize.addPass(llvm::LoopUnrollPass(options));
            optimize.run(function, analyses);
            for (auto &block : function) {
                auto *instruction = block.getTerminator();
                auto found = originals.find(
                    instruction->getMetadata(llvm::LLVMContext::MD_loop));
                if (found != originals.end())
                    instruction->setMetadata(llvm::LLVMContext::MD_loop,
                                             found->second);
            }
            analyses.invalidate(function, llvm::PreservedAnalyses::none());
            llvm::FunctionPassManager scalarize;
            scalarize.addPass(llvm::SROAPass(llvm::SROAOptions::PreserveCFG));
            scalarize.addPass(llvm::SimplifyCFGPass{});
            scalarize.run(function, analyses);
        }
        return changed ? llvm::PreservedAnalyses::none()
                       : llvm::PreservedAnalyses::all();
    }

  private:
    static bool requests_unroll(llvm::MDNode *id) {
        if (!id)
            return false;
        bool requested = false;
        for (unsigned i = 1; i < id->getNumOperands(); ++i) {
            auto *node =
                llvm::dyn_cast_or_null<llvm::MDNode>(id->getOperand(i));
            auto *name = node && node->getNumOperands()
                             ? llvm::dyn_cast_or_null<llvm::MDString>(
                                   node->getOperand(0))
                             : nullptr;
            if (!name)
                continue;
            auto key = name->getString();
            if (key == "llvm.loop.unroll.disable" ||
                key == "llvm.loop.unroll.count")
                return false;
            requested |= key == "llvm.loop.unroll.enable" ||
                         key == "llvm.loop.unroll.full";
        }
        return requested;
    }
};

} // namespace cumetal::ir
