#pragma once
#include <boost/regex/icu.hpp>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace inference {
    using TokenId = std::size_t;

    struct TokenList {
        std::vector<TokenId> tokens;
    };

    struct MergePair {
        std::string left;
        std::string right;
        bool operator==(const MergePair&) const = default;
    };

    struct MergePairHasher {
        // implemented in cpp so we don't include boost::hash here
        std::size_t operator()(const MergePair& pair) const noexcept;
    };

    struct TokenizerVocab {
        std::unordered_map<std::string, TokenId> word_to_id;
        std::unordered_map<TokenId, std::string> id_to_word;
    };

    class Tokenizer {
    public:
        [[nodiscard]] static Tokenizer parse_tokenizer(const std::filesystem::path& tokenizer_path,
                                                       const std::filesystem::path& tokenizer_config_path);

        [[nodiscard]] TokenList tokenize(std::string_view prompt);
        [[nodiscard]] std::string decode(const TokenList& token_list);

        [[nodiscard]] std::vector<std::string> pre_tokenize(const std::string& prompt) const;

    private:
        // TODO: can we switch to std::string_view to not have duplicate allocations of the same string?
        TokenizerVocab vocab_;
        std::unordered_map<MergePair, std::size_t, MergePairHasher> merges_; // merge pair to rank, 0 is highest rank
        boost::u32regex pretokenizer_regex_;
        // todo special token
    };
} // namespace inference
