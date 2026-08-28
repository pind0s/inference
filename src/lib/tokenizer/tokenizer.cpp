#include "tokenizer.hpp"
#include "types/token.hpp"
#include "util/files.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <nlohmann/json.hpp>
#include <ranges>
#include <string>
#include <unicode/normalizer2.h>
#include <unicode/unistr.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace inference::tokenizer {
    TokenizerVocab Tokenizer::parse_vocab(const nlohmann::json& tokenizer) {
        TokenizerVocab result;
        result.word_to_id = tokenizer.at("model").at("vocab").get<decltype(TokenizerVocab::word_to_id)>();

        result.id_to_word.reserve(result.word_to_id.size());
        for (const auto& [word, token_id] : result.word_to_id) {
            result.id_to_word.emplace(token_id, word);
        }

        return result;
    }

    std::unordered_map<MergePair, std::size_t, MergePairHasher> Tokenizer::parse_merges(const nlohmann::json& tokenizer) {
        auto merges = tokenizer.at("model").at("merges").get<std::vector<std::array<std::string, 2>>>();

        std::unordered_map<MergePair, std::size_t, MergePairHasher> result;
        result.reserve(merges.size());
        for (auto&& [index, merge] : merges | std::views::enumerate) {
            result.emplace(MergePair{ .left = std::move(merge[0]), .right = std::move(merge[1]) }, index);
        }
        return result;
    }

    TokenizerVocab Tokenizer::parse_special_tokens(const nlohmann::json& tokenizer) {
        TokenizerVocab result;

        for (const auto& token : tokenizer.at("added_tokens")) {
            const auto token_id = token.at("id").get<types::TokenId>();
            const auto token_text = token.at("content").get<std::string>();

            result.word_to_id.emplace(token_text, token_id);
            result.id_to_word.emplace(token_id, token_text);
        }

        return result;
    }

    std::string Tokenizer::parse_pretokenizer_regex(const nlohmann::json& tokenizer) {
        auto pattern = tokenizer.at("pre_tokenizer").at("pretokenizers").at(0).at("pattern").at("Regex").get<std::string>();

        // boost requires long names for Unicode categories
        pattern = boost::regex_replace(pattern, boost::regex(R"(\\p\{L\})"), R"(\\p{Letter})");
        pattern = boost::regex_replace(pattern, boost::regex(R"(\\p\{N\})"), R"(\\p{Number})");

        return pattern;
    }

    boost::regex Tokenizer::make_special_token_regex(const TokenizerVocab& special_tokens) {
        static const boost::regex pipe_regex{ R"(\|)" };

        std::vector<std::string> tokens;
        for (const auto& token : special_tokens.word_to_id | std::views::keys) {
            tokens.push_back(boost::regex_replace(token, pipe_regex, R"(\\|)"));
        }

        std::string pattern;
        for (const auto& token : tokens) {
            if (!pattern.empty()) {
                pattern += '|';
            }
            pattern += token;
        }
        return boost::regex{ pattern };
    }

    auto Tokenizer::split_special_tokens(const std::string& text, const boost::regex& special_token_regex) {
        using Iterator = boost::sregex_token_iterator;

        // -1 returns text between regex matches, 0 returns regex match itself. don't ask me, i have no idea.
        return std::ranges::subrange{ Iterator{ text.begin(), text.end(), special_token_regex, { -1, 0 } }, Iterator{} };
    }

    TokenList Tokenizer::tokenize_text(const std::string& text) const {
        const auto normalized = normalize_nfc(text);
        const auto pieces = pre_tokenize(normalized);

        TokenList ids;
        for (const auto& piece : pieces) {
            for (const auto& symbol : merge_bpe(byte_level_encode(piece))) {
                ids.push_back(vocab_.word_to_id.at(symbol));
            }
        }

        return ids;
    }

    std::string Tokenizer::normalize_nfc(std::string_view prompt) {
        UErrorCode error = U_ZERO_ERROR;
        const auto* normalizer = icu::Normalizer2::getNFCInstance(error);

        const auto source = icu::UnicodeString::fromUTF8(prompt);
        icu::UnicodeString normalized;
        normalizer->normalize(source, normalized, error);

        if (U_FAILURE(error) != 0) {
            throw std::runtime_error("Error normalizing text");
        }

        std::string result;
        normalized.toUTF8String(result);
        return result;
    }

    std::vector<std::string> Tokenizer::pre_tokenize(const std::string& prompt) const {
        auto begin = boost::make_u32regex_token_iterator(prompt, pretokenizer_regex_);
        auto end = decltype(begin){}; // default constructed iterator type represents end
        return { begin, end };
    }

    std::vector<std::string> Tokenizer::byte_level_encode(const std::string_view piece) {
        static const auto byte_to_codepoint = byte_to_codepoint_table();

        std::vector<std::string> symbols;
        for (const unsigned char byte : piece) {
            symbols.push_back(byte_to_codepoint[byte]);
        }
        return symbols;
    }

    std::vector<std::string> Tokenizer::merge_bpe(std::vector<std::string> token_list) const {
        const auto make_merge_pair = [](const std::string& lhs, const std::string& rhs) {
            return MergePair{ .left = lhs, .right = rhs };
        };

        while (token_list.size() > 1) {
            // find all mergeable pairs
            auto candidates = token_list | std::views::pairwise_transform(make_merge_pair) |
                              std::views::filter([this](const MergePair& candidate) { return merges_.contains(candidate); });

            // nothing to merge, exit
            if (candidates.empty()) {
                break;
            }

            // find the best pair
            const auto best =
                std::ranges::min_element(candidates, [this](const MergePair& lhs, const MergePair& rhs) { return merges_.at(lhs) < merges_.at(rhs); });

            // merge
            const auto [left, right] = *best;
            for (std::size_t i = 0; i + 1 < token_list.size(); ++i) {
                if (token_list[i] == left && token_list[i + 1] == right) {
                    token_list[i].append(token_list[i + 1]);
                    token_list.erase(token_list.begin() + i + 1);
                    break;
                }
            }
        }

        return token_list;
    }

    std::unordered_map<std::string, std::byte> Tokenizer::codepoint_to_byte_map() {
        std::unordered_map<std::string, std::byte> result;
        const auto byte_encoder = byte_to_codepoint_table();
        result.reserve(byte_encoder.size());

        for (std::size_t byte = 0; byte < byte_encoder.size(); ++byte) {
            result.emplace(byte_encoder[byte], static_cast<std::byte>(byte));
        }

        return result;
    }

    // good reference: https://github.com/ml-rust/splintr/blob/main/docs/bytelevel_bpe.md
    std::array<std::string, Tokenizer::BYTE_VOCAB_SIZE> Tokenizer::byte_to_codepoint_table() {
        std::array<std::string, BYTE_VOCAB_SIZE> table;

        UChar32 next_remapped = U'\u0100';
        for (const auto byte : std::views::iota(0uz, table.size())) {
            const bool unchanged = (byte >= 0x21 && byte <= 0x7E) || (byte >= 0xA1 && byte <= 0xAC) || (byte >= 0xAE && byte <= 0xFF);

            const auto mapped_codepoint = unchanged ? static_cast<UChar32>(byte) : next_remapped++;
            table[byte] = codepoint_to_utf8(mapped_codepoint);
        }

        return table;
    }

    std::string Tokenizer::codepoint_to_utf8(const UChar32 codepoint) {
        icu::UnicodeString unicode;
        unicode.append(codepoint);

        std::string result;
        unicode.toUTF8String(result);
        return result;
    }

    Tokenizer::Tokenizer(const std::filesystem::path& model_path) {
        constexpr std::string_view tokenizer_file = "tokenizer.json";
        const auto file = util::read_file(model_path / tokenizer_file);
        const auto tokenizer_json = nlohmann::json::parse(file);

        pretokenizer_regex_ = boost::make_u32regex(parse_pretokenizer_regex(tokenizer_json));
        vocab_ = parse_vocab(tokenizer_json);
        merges_ = parse_merges(tokenizer_json);
        special_tokens_ = parse_special_tokens(tokenizer_json);
        special_token_regex_ = make_special_token_regex(special_tokens_);
    }

    TokenList Tokenizer::tokenize(const std::string& prompt) const {
        TokenList ids;

        for (const auto& match : split_special_tokens(prompt, special_token_regex_)) {
            const auto piece = match.str();

            if (piece.empty()) {
                continue;
            }

            if (special_tokens_.word_to_id.contains(piece)) {
                ids.push_back(special_tokens_.word_to_id.at(piece));
            } else {
                ids.insert_range(ids.end(), tokenize_text(piece));
            }
        }

        return ids;
    }

    std::string Tokenizer::decode(const TokenList& token_list) const {
        static const auto codepoint_to_byte = codepoint_to_byte_map();
        std::string decoded;

        for (const auto token_id : token_list) {
            if (special_tokens_.id_to_word.contains(token_id)) {
                decoded.append(special_tokens_.id_to_word.at(token_id));
                continue;
            }

            const auto& token = vocab_.id_to_word.at(token_id);
            const auto unicode = icu::UnicodeString::fromUTF8(token);

            for (std::int32_t offset = 0; offset < unicode.length();) {
                const auto codepoint = unicode.char32At(offset);
                const auto encoded_byte = codepoint_to_utf8(codepoint);

                const auto byte = codepoint_to_byte.find(encoded_byte);
                if (byte == codepoint_to_byte.end()) {
                    throw std::runtime_error("token contains a character outside the byte-level encoding");
                }

                decoded.push_back(static_cast<char>(byte->second));
                offset += U16_LENGTH(codepoint);
            }
        }

        return decoded;
    }

} // namespace inference::tokenizer
