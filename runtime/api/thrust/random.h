#pragma once

#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define CUMETAL_THRUST_RANDOM_HD __host__ __device__
#else
#define CUMETAL_THRUST_RANDOM_HD
#endif

namespace thrust {

class default_random_engine {
public:
    using result_type = unsigned int;

    CUMETAL_THRUST_RANDOM_HD explicit default_random_engine(result_type seed_value = 1u) {
        seed(seed_value);
    }

    CUMETAL_THRUST_RANDOM_HD void seed(result_type value = 1u) {
        state_ = value % modulus;
        if (state_ == 0u) state_ = 1u;
    }

    CUMETAL_THRUST_RANDOM_HD result_type operator()() {
        state_ = static_cast<result_type>(
            (static_cast<unsigned long long>(state_) * multiplier) % modulus);
        return state_;
    }

    CUMETAL_THRUST_RANDOM_HD static constexpr result_type min() { return 1u; }
    CUMETAL_THRUST_RANDOM_HD static constexpr result_type max() { return modulus - 1u; }

private:
    static constexpr result_type multiplier = 48271u;
    static constexpr result_type modulus = 2147483647u;
    result_type state_ = 1u;
};

template <typename T = int>
class uniform_int_distribution {
public:
    CUMETAL_THRUST_RANDOM_HD explicit uniform_int_distribution(
        T lower = 0, T upper = std::numeric_limits<T>::max())
        : lower_(lower), upper_(upper) {}

    template <typename Engine>
    CUMETAL_THRUST_RANDOM_HD T operator()(Engine& engine) const {
        const unsigned long long span =
            static_cast<unsigned long long>(upper_) -
            static_cast<unsigned long long>(lower_) + 1ull;
        const unsigned long long sample =
            (static_cast<unsigned long long>(engine()) << 31u) ^ engine();
        return static_cast<T>(static_cast<unsigned long long>(lower_) +
                              (span == 0ull ? sample : sample % span));
    }

private:
    T lower_;
    T upper_;
};

template <typename T = double>
class uniform_real_distribution {
public:
    CUMETAL_THRUST_RANDOM_HD explicit uniform_real_distribution(T lower = T(0),
                                                                 T upper = T(1))
        : lower_(lower), upper_(upper) {}

    template <typename Engine>
    CUMETAL_THRUST_RANDOM_HD T operator()(Engine& engine) const {
        const T unit = static_cast<T>(engine() - Engine::min()) /
                       static_cast<T>(Engine::max() - Engine::min() + 1u);
        return lower_ + (upper_ - lower_) * unit;
    }

private:
    T lower_;
    T upper_;
};

namespace random {
using default_random_engine = ::thrust::default_random_engine;
template <typename T = int>
using uniform_int_distribution = ::thrust::uniform_int_distribution<T>;
template <typename T = double>
using uniform_real_distribution = ::thrust::uniform_real_distribution<T>;
}

}

#undef CUMETAL_THRUST_RANDOM_HD
