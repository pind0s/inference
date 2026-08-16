#pragma once
#include "types/token.hpp"
#include <array>
#include <boost/regex/icu.hpp>
#include <filesystem>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace inference::tokenizer {
    using TokenList = std::vector<types::TokenId>;

    struct MergePair {
        std::string left;
        std::string right;
        bool operator==(const MergePair&) const = default;
    };

    struct MergePairHasher {
        std::size_t operator()(const MergePair& pair) const noexcept {
            const auto left = std::hash<std::string>{}(pair.left);
            const auto right = std::hash<std::string>{}(pair.right);
            return left ^ (right << 1uz);
        }
    };

    struct TokenizerVocab {
        std::unordered_map<std::string, types::TokenId> word_to_id;
        std::unordered_map<types::TokenId, std::string> id_to_word;
    };

    class Tokenizer {
    public:
        explicit Tokenizer(const std::filesystem::path& model_path);

        [[nodiscard]] TokenList tokenize(const std::string& prompt) const;
        [[nodiscard]] std::string decode(const TokenList& token_list) const;

    private:
        static constexpr std::size_t BYTE_VOCAB_SIZE = 256;

        // todo should we even make these static? just make them return void and directly init the stuff.
        [[nodiscard]] static TokenizerVocab parse_vocab(const nlohmann::json& tokenizer);
        [[nodiscard]] static std::unordered_map<MergePair, std::size_t, MergePairHasher> parse_merges(const nlohmann::json& tokenizer);
        [[nodiscard]] static TokenizerVocab parse_special_tokens(const nlohmann::json& tokenizer);
        [[nodiscard]] static std::string parse_pretokenizer_regex(const nlohmann::json& tokenizer);
        [[nodiscard]] static boost::regex make_special_token_regex(const TokenizerVocab& special_tokens);

        [[nodiscard]] static auto split_special_tokens(const std::string& text, const boost::regex& special_token_regex);
        [[nodiscard]] TokenList tokenize_text(const std::string& text) const;
        [[nodiscard]] static std::string normalize_nfc(std::string_view prompt);
        [[nodiscard]] std::vector<std::string> pre_tokenize(const std::string& prompt) const;
        [[nodiscard]] static std::vector<std::string> byte_level_encode(std::string_view piece);
        [[nodiscard]] std::vector<std::string> merge_bpe(std::vector<std::string> token_list) const;

        [[nodiscard]] static std::unordered_map<std::string, std::byte> codepoint_to_byte_map();
        [[nodiscard]] static std::array<std::string, BYTE_VOCAB_SIZE> byte_to_codepoint_table();
        [[nodiscard]] static std::string codepoint_to_utf8(UChar32 codepoint);

        TokenizerVocab vocab_;
        TokenizerVocab special_tokens_;
        boost::regex special_token_regex_;
        std::unordered_map<MergePair, std::size_t, MergePairHasher> merges_;
        boost::u32regex pretokenizer_regex_;
    };
} // namespace inference::tokenizer
