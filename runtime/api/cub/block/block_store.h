#pragma once
// CuMetal CUB shim: BlockStore — cooperative block-wide data storing.

namespace cub {

enum BlockStoreAlgorithm {
    BLOCK_STORE_DIRECT,
    BLOCK_STORE_VECTORIZE,
    BLOCK_STORE_TRANSPOSE,
    BLOCK_STORE_WARP_TRANSPOSE,
    BLOCK_STORE_STRIPED
};

template <typename T, int BLOCK_DIM_X, int ITEMS_PER_THREAD,
          BlockStoreAlgorithm ALGORITHM = BLOCK_STORE_DIRECT,
          int BLOCK_DIM_Y = 1, int BLOCK_DIM_Z = 1>
class BlockStore {
    static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

public:
    struct TempStorage {};

    explicit BlockStore(TempStorage&) : linear_tid_(0) {}
    BlockStore(TempStorage&, int linear_tid) : linear_tid_(linear_tid) {}

    // Store a full tile
    void Store(T* block_ptr, T (&items)[ITEMS_PER_THREAD]) {
        for (int i = 0; i < ITEMS_PER_THREAD; i++)
            block_ptr[linear_tid_ * ITEMS_PER_THREAD + i] = items[i];
    }

    // Store a partial tile
    void Store(T* block_ptr, T (&items)[ITEMS_PER_THREAD], int valid_items) {
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int idx = linear_tid_ * ITEMS_PER_THREAD + i;
            if (idx < valid_items) block_ptr[idx] = items[i];
        }
    }

    // Striped store
    void StoreStriped(T* block_ptr, T (&items)[ITEMS_PER_THREAD]) {
        for (int i = 0; i < ITEMS_PER_THREAD; i++)
            block_ptr[linear_tid_ + i * BLOCK_THREADS] = items[i];
    }

private:
    int linear_tid_;
};

} // namespace cub
