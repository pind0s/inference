#pragma once
#include <boost/regex/icu.hpp>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace inference {
    using TokenId = std::size_t;
    using TokenList = std::vector<TokenId>;

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
        [[nodiscard]] static Tokenizer load_tokenizer(const std::filesystem::path& tokenizer_path,
                                                      const std::filesystem::path& tokenizer_config_path);

        [[nodiscard]] TokenList tokenize(const std::string& prompt) const;
        [[nodiscard]] std::string decode(const TokenList& token_list) const;

    private:
        [[nodiscard]] std::vector<std::string> pre_tokenize(const std::string& prompt) const;
        [[nodiscard]] std::vector<std::string> merge_bpe(std::vector<std::string> token_list) const;

        // TODO(pind0s): can we switch to std::string_view to not have duplicate allocations of the same string?
        TokenizerVocab vocab_;
        // merge pair to rank, 0 is highest rank
        std::unordered_map<MergePair, std::size_t, MergePairHasher> merges_;
        boost::u32regex pretokenizer_regex_;
        // TODO(pind0s): special token
    };
} // namespace inference
