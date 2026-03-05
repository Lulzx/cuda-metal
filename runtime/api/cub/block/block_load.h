#pragma once
// CuMetal CUB shim: BlockLoad — cooperative block-wide data loading.

#include <cstring>

namespace cub {

enum BlockLoadAlgorithm {
    BLOCK_LOAD_DIRECT,
    BLOCK_LOAD_VECTORIZE,
    BLOCK_LOAD_TRANSPOSE,
    BLOCK_LOAD_WARP_TRANSPOSE,
    BLOCK_LOAD_STRIPED
};

template <typename InputT, int BLOCK_DIM_X, int ITEMS_PER_THREAD,
          BlockLoadAlgorithm ALGORITHM = BLOCK_LOAD_DIRECT,
          int BLOCK_DIM_Y = 1, int BLOCK_DIM_Z = 1>
class BlockLoad {
    static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    static constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

public:
    struct TempStorage {};

    explicit BlockLoad(TempStorage&) : linear_tid_(0) {}
    BlockLoad(TempStorage&, int linear_tid) : linear_tid_(linear_tid) {}

    // Load a full tile
    void Load(const InputT* block_ptr, InputT (&items)[ITEMS_PER_THREAD]) {
        for (int i = 0; i < ITEMS_PER_THREAD; i++)
            items[i] = block_ptr[linear_tid_ * ITEMS_PER_THREAD + i];
    }

    // Load a partial tile with default value for out-of-bounds
    void Load(const InputT* block_ptr, InputT (&items)[ITEMS_PER_THREAD],
              int valid_items, InputT oob_default = InputT{}) {
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int idx = linear_tid_ * ITEMS_PER_THREAD + i;
            items[i] = (idx < valid_items) ? block_ptr[idx] : oob_default;
        }
    }

    // Striped load
    void LoadStriped(const InputT* block_ptr, InputT (&items)[ITEMS_PER_THREAD]) {
        for (int i = 0; i < ITEMS_PER_THREAD; i++)
            items[i] = block_ptr[linear_tid_ + i * BLOCK_THREADS];
    }

private:
    int linear_tid_;
};

} // namespace cub
