#pragma once
#include "backend/cpu/bf16.hpp"
#include "tensor/tensor.hpp"
#include "types/dtype.hpp"
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <format>
#include <string>
#include <utility>

namespace test {
    class WithinRel final : public Catch::Matchers::MatcherGenericBase {
    public:
        WithinRel(const inference::Tensor& expected, const float epsilon, const float margin = 0.0f)
            : expected_(expected),
              epsilon_(epsilon),
              margin_(margin) { }

        [[nodiscard]] bool match(const inference::Tensor& actual) const {
            using namespace inference;
            mismatch_.clear();
            if (!actual.is_cpu() || !expected_.is_cpu()) {
                return fail("both tensors must be on CPU");
            }
            if (actual.dtype() != types::DType::BF16 || expected_.dtype() != types::DType::BF16) {
                return fail("both tensors must have BF16 dtype");
            }
            const auto actual_shape = actual.shape();
            const auto expected_shape = expected_.shape();
            if (actual_shape.rank() != expected_shape.rank()) {
                return fail(std::format("rank mismatch: actual {}, expected {}", actual_shape.rank(), expected_shape.rank()));
            }
            for (int dim = 0; dim < expected_shape.rank(); ++dim) {
                if (actual_shape.dim(dim) != expected_shape.dim(dim)) {
                    return fail(
                        std::format("shape mismatch at dimension {}: actual {}, expected {}", dim, actual_shape.dim(dim), expected_shape.dim(dim)));
                }
            }

            const auto actual_values = actual.view<cpu::bf16_t>();
            const auto expected_values = expected_.view<cpu::bf16_t>();
            for (std::size_t i = 0; i < expected_values.size(); ++i) {
                const float value = actual_values[i];
                const float target = expected_values[i];
                if (!Catch::Matchers::WithinRel(target, epsilon_).match(value) && !Catch::Matchers::WithinAbs(target, margin_).match(value)) {
                    return fail(std::format("first mismatch at flat index {}: actual {}, expected {}", i, value, target));
                }
            }
            return true;
        }

    private:
        [[nodiscard]] std::string describe() const override {
            auto description = std::format("matches the expected BF16 tensor within relative tolerance {}", epsilon_);
            if (margin_ != 0.0f) {
                description += std::format(" or absolute tolerance {}", margin_);
            }
            if (!mismatch_.empty()) {
                description += "; " + mismatch_;
            }
            return description;
        }

        bool fail(std::string message) const {
            mismatch_ = std::move(message);
            return false;
        }

        inference::Tensor expected_;
        float epsilon_;
        float margin_;
        mutable std::string mismatch_;
    };
} // namespace test
