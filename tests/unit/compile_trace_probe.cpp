#include "cumetal/common/compile_trace.h"

#include <barrier>
#include <cstdlib>
#include <string_view>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 64;
    const std::string_view mode = argv[1];
    if (mode == "nested") {
        cumetal::common::CompileTrace outer("outer", 64);
        cumetal::common::CompileTrace inner("inner", 8);
        return 0;  // Both spans must close on early return.
    }
    if (mode == "concurrent") {
        constexpr int count = 8;
        std::barrier ready(count);
        std::vector<std::thread> threads;
        for (int i = 0; i < count; ++i) {
            threads.emplace_back([&] {
                cumetal::common::CompileTrace trace("parallel", 4);
                ready.arrive_and_wait();
            });
        }
        for (auto& thread : threads) thread.join();
        return 0;
    }
    if (mode == "interrupted") {
        cumetal::common::CompileTrace trace("interrupted", 32);
        std::_Exit(42);  // No stack unwinding: the flushed begin must survive.
    }
    return 64;
}
