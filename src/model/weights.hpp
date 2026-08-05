#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "tensor/tensor.hpp"
#include "util/move_only.hpp"

namespace inference {
    class Weights : util::MoveOnly {
    public:
        Weights(): tensors_{std::make_shared<TensorMap>()} { }

        void insert(const std::string& name, Tensor tensor) {
            tensors_->emplace(full_name(name), std::move(tensor));
        }

        [[nodiscard]] Tensor take(const std::string_view name) {
            const auto name_with_prefix = full_name(name);
            const auto node = tensors_->extract(name_with_prefix);
            if (node.empty()) {
                throw std::invalid_argument("No weights found for '" + name_with_prefix + "'");
            }
            return std::move(node.mapped());
        }

        [[nodiscard]] Weights scope(const std::string_view name) const {
            auto prefix = full_name(name);
            prefix.push_back('.');
            return Weights{tensors_, std::move(prefix)};
        }

        [[nodiscard]] Weights scope(const std::size_t index) const {
            return scope(std::to_string(index));
        }

        void expect_empty() const {
            if (!tensors_->empty()) {
                throw std::runtime_error("Unexpected leftover tensor '" + tensors_->begin()->first + "'");
            }
        }

    private:
        using TensorMap = std::unordered_map<std::string, Tensor>;

        Weights(std::shared_ptr<TensorMap> tensors, std::string prefix): tensors_{std::move(tensors)}, prefix_{std::move(prefix)} { }

        [[nodiscard]] std::string full_name(const std::string_view name) const {
            auto result = prefix_;
            result.append(name);
            return result;
        }

        std::shared_ptr<TensorMap> tensors_;
        std::string prefix_;
    };
} // namespace inference
