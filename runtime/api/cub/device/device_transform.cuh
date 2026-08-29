#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace cub {
namespace detail {

template <typename Inputs, typename Op, std::size_t... I>
auto invoke_at(const Inputs& inputs, Op& op, std::size_t index,
               std::index_sequence<I...>) {
    return op(std::get<I>(inputs)[index]...);
}

template <typename Outputs, typename Result, std::size_t... I>
void store_tuple(Outputs& outputs, Result&& result, std::size_t index,
                 std::index_sequence<I...>) {
    ((std::get<I>(outputs)[index] = std::get<I>(result)), ...);
}

template <typename T>
struct is_tuple : std::false_type {};
template <typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};

}

struct DeviceTransform {
    template <typename... Inputs, typename Output, typename Size, typename Op>
    static cudaError_t Transform(std::tuple<Inputs...> inputs, Output output,
                                 Size count, Op op, cudaStream_t = nullptr) {
        for (Size i = 0; i < count; ++i) {
            auto result = detail::invoke_at(inputs, op, static_cast<std::size_t>(i),
                                            std::index_sequence_for<Inputs...>{});
            if constexpr (detail::is_tuple<std::decay_t<Output>>::value) {
                detail::store_tuple(output, result, static_cast<std::size_t>(i),
                                    std::make_index_sequence<
                                        std::tuple_size_v<std::decay_t<Output>>>{});
            } else {
                output[i] = result;
            }
        }
        return cudaSuccess;
    }
};

}
