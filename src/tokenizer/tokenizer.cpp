#include <boost/container_hash/hash.hpp>
#include <boost/regex/icu.hpp>
#include <nlohmann/json.hpp>
#include <print>

#include "tokenizer.hpp"
#include "util/files.hpp"

namespace inference {
    namespace details {
        // todo parse_special_tokens
        [[nodiscard]]
        std::string parse_regex(const nlohmann::json& tokenizer_json) {
            auto pattern =
                tokenizer_json.at("pre_tokenizer").at("pretokenizers").at(0).at("pattern").at("Regex").get<std::string>();

            // boost requires long names for Unicode categories
            pattern = boost::regex_replace(pattern, boost::regex(R"(\\p\{L\})"), R"(\\p{Letter})");
            pattern = boost::regex_replace(pattern, boost::regex(R"(\\p\{N\})"), R"(\\p{Number})");

            return pattern;
        }

        [[nodiscard]]
        TokenizerVocab parse_vocab(const nlohmann::json& tokenizer_json) {
            TokenizerVocab result;
            result.word_to_id = tokenizer_json.at("model").at("vocab").get<decltype(TokenizerVocab::word_to_id)>();

            result.id_to_word.reserve(result.word_to_id.size());
            for (const auto& [word, token_id] : result.word_to_id) {
                result.id_to_word.emplace(token_id, word);
            }

            return result;
        }

        [[nodiscard]]
        std::unordered_map<MergePair, std::size_t, MergePairHasher> parse_merges(const nlohmann::json& tokenizer_json) {
            auto merges = tokenizer_json.at("model").at("merges").get<std::vector<std::array<std::string, 2>>>();

            std::unordered_map<MergePair, std::size_t, MergePairHasher> result;
            result.reserve(merges.size());
            for (auto&& [index, merge] : merges | std::views::enumerate) {
                result.emplace(MergePair{.left = std::move(merge[0]), .right = std::move(merge[1])}, index);
            }
            return result;
        }

    } // namespace details

    std::size_t MergePairHasher::operator()(const MergePair& pair) const noexcept {
        std::size_t seed = 0;
        boost::hash_combine(seed, pair.left);
        boost::hash_combine(seed, pair.right);
        return seed;
    }

    Tokenizer Tokenizer::parse_tokenizer(const std::filesystem::path& tokenizer_path,
                                         const std::filesystem::path& tokenizer_config_path) {

        auto tokenizer_file = util::read_file(tokenizer_path);
        auto config_file = util::read_file(tokenizer_config_path);

        if (!tokenizer_file.has_value()) {
            const auto error = std::format("failed to read {}", tokenizer_path.string());
            throw std::runtime_error(error);
        }

        if (!config_file.has_value()) {
            const auto error = std::format("failed to read {}", tokenizer_config_path.string());
            throw std::runtime_error(error);
        }

        auto tokenizer_json = nlohmann::json::parse(tokenizer_file.value());
        auto config_json = nlohmann::json::parse(config_file.value());

        Tokenizer tokenizer;
        tokenizer.pretokenizer_regex_ = boost::make_u32regex(details::parse_regex(tokenizer_json));
        tokenizer.vocab_ = details::parse_vocab(tokenizer_json);
        tokenizer.merges_ = details::parse_merges(tokenizer_json);

        return tokenizer;
    }

    TokenList Tokenizer::tokenize(std::string_view prompt) {
        return {};
    }

    std::string Tokenizer::decode(const TokenList& token_list) {
        return {};
    }

    std::vector<std::string> Tokenizer::pre_tokenize(const std::string& prompt) const {
        auto begin = boost::make_u32regex_token_iterator(prompt, pretokenizer_regex_);
        auto end = decltype(begin){}; // default constructed iterator type represents end
        return {begin, end};
    }
} // namespace inference
