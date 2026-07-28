#include <algorithm>
#include <boost/container_hash/hash.hpp>
#include <boost/regex.hpp>
#include <boost/regex/icu.hpp>
#include <nlohmann/json.hpp>
#include <unicode/normalizer2.h>
#include <unicode/unistr.h>

#include "tokenizer.hpp"
#include "util/files.hpp"

namespace inference {
    // todo probably make this static and add them to Tokenizer class
    namespace {
        constexpr std::size_t BYTE_VOCAB_SIZE = 256;

        [[nodiscard]]
        TokenizerVocab parse_special_tokens(const nlohmann::json& tokenizer) {
            TokenizerVocab result;

            for (const auto& token : tokenizer.at("added_tokens")) {
                const auto token_id = token.at("id").get<TokenId>();
                const auto token_text = token.at("content").get<std::string>();

                result.word_to_id.emplace(token_text, token_id);
                result.id_to_word.emplace(token_id, token_text);
            }

            return result;
        }

        [[nodiscard]]
        boost::regex make_special_token_regex(const TokenizerVocab& special_tokens) {
            static const boost::regex pipe_regex{R"(\|)"};

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

            return boost::regex{pattern};
        }

        [[nodiscard]]
        std::string parse_pretokenizer_regex(const nlohmann::json& tokenizer) {
            auto pattern =
                tokenizer.at("pre_tokenizer").at("pretokenizers").at(0).at("pattern").at("Regex").get<std::string>();

            // boost requires long names for Unicode categories
            pattern = boost::regex_replace(pattern, boost::regex(R"(\\p\{L\})"), R"(\\p{Letter})");
            pattern = boost::regex_replace(pattern, boost::regex(R"(\\p\{N\})"), R"(\\p{Number})");

            return pattern;
        }

        [[nodiscard]]
        TokenizerVocab parse_vocab(const nlohmann::json& tokenizer) {
            TokenizerVocab result;
            result.word_to_id = tokenizer.at("model").at("vocab").get<decltype(TokenizerVocab::word_to_id)>();

            result.id_to_word.reserve(result.word_to_id.size());
            for (const auto& [word, token_id] : result.word_to_id) {
                result.id_to_word.emplace(token_id, word);
            }

            return result;
        }

        [[nodiscard]]
        std::unordered_map<MergePair, std::size_t, MergePairHasher> parse_merges(const nlohmann::json& tokenizer) {
            auto merges = tokenizer.at("model").at("merges").get<std::vector<std::array<std::string, 2>>>();

            std::unordered_map<MergePair, std::size_t, MergePairHasher> result;
            result.reserve(merges.size());
            for (auto&& [index, merge] : merges | std::views::enumerate) {
                result.emplace(MergePair{.left = std::move(merge[0]), .right = std::move(merge[1])}, index);
            }
            return result;
        }

        [[nodiscard]]
        std::string codepoint_to_utf8(const UChar32 codepoint) {
            icu::UnicodeString unicode;
            unicode.append(codepoint);

            std::string result;
            unicode.toUTF8String(result);
            return result;
        }

        // good reference: https://github.com/ml-rust/splintr/blob/main/docs/bytelevel_bpe.md
        [[nodiscard]]
        std::array<std::string, BYTE_VOCAB_SIZE> byte_to_codepoint_table() {
            std::array<std::string, BYTE_VOCAB_SIZE> table;

            UChar32 next_remapped = U'\u0100';
            // TODO(pind0s): std::ranges::views::indices c++26 only :(
            for (const auto byte : std::views::iota(std::size_t{0}, table.size())) {
                const bool unchanged =
                    (byte >= 0x21 && byte <= 0x7E) || (byte >= 0xA1 && byte <= 0xAC) || (byte >= 0xAE && byte <= 0xFF);

                const auto mapped_codepoint = unchanged ? static_cast<UChar32>(byte) : next_remapped++;
                table[byte] = codepoint_to_utf8(mapped_codepoint);
            }

            return table;
        }

        [[nodiscard]]
        std::unordered_map<std::string, std::byte> codepoint_to_byte_map() {
            std::unordered_map<std::string, std::byte> result;
            const auto byte_encoder = byte_to_codepoint_table();
            result.reserve(byte_encoder.size());

            for (std::size_t byte = 0; byte < byte_encoder.size(); ++byte) {
                result.emplace(byte_encoder[byte], static_cast<std::byte>(byte));
            }

            return result;
        }

        [[nodiscard]]
        std::string normalize_nfc(std::string_view prompt) {
            UErrorCode error = U_ZERO_ERROR;
            const auto* normalizer = icu::Normalizer2::getNFCInstance(error);

            const auto source = icu::UnicodeString::fromUTF8(prompt);
            icu::UnicodeString normalized;
            normalizer->normalize(source, normalized, error);

            if (error > U_ZERO_ERROR) {
                throw std::runtime_error("Error normalizing text");
            }

            std::string result;
            normalized.toUTF8String(result);
            return result;
        }

        [[nodiscard]]
        std::vector<std::string> byte_level_encode(const std::string_view piece) {
            static const auto byte_to_codepoint = byte_to_codepoint_table();

            std::vector<std::string> symbols;
            for (const unsigned char byte : piece) {
                symbols.push_back(byte_to_codepoint[byte]);
            }
            return symbols;
        }

        [[nodiscard]]
        auto split_special_tokens(const std::string& text, const boost::regex& special_token_regex) {
            using Iterator = boost::sregex_token_iterator;

            // -1 returns text between regex matches, 0 returns regex match itself. don't ask me, i have no idea.
            return std::ranges::subrange{Iterator{text.begin(), text.end(), special_token_regex, {-1, 0}}, Iterator{}};
        }

    } // namespace

    std::size_t MergePairHasher::operator()(const MergePair& pair) const noexcept {
        std::size_t seed = 0;
        boost::hash_combine(seed, pair.left);
        boost::hash_combine(seed, pair.right);
        return seed;
    }

    Tokenizer Tokenizer::load_tokenizer(const std::filesystem::path& tokenizer_path,
                                        const std::filesystem::path& tokenizer_config_path) {

        auto tokenizer_file = util::read_file(tokenizer_path);
        if (!tokenizer_file) {
            const auto error = std::format("failed to read {}", tokenizer_path.string());
            throw std::runtime_error(error);
        }

        auto config_file = util::read_file(tokenizer_config_path);
        if (!config_file) {
            const auto error = std::format("failed to read {}", tokenizer_config_path.string());
            throw std::runtime_error(error);
        }

        const auto tokenizer_json = nlohmann::json::parse(tokenizer_file.value());
        const auto config_json = nlohmann::json::parse(config_file.value());

        Tokenizer result;
        result.pretokenizer_regex_ = boost::make_u32regex(parse_pretokenizer_regex(tokenizer_json));
        result.vocab_ = parse_vocab(tokenizer_json);
        result.merges_ = parse_merges(tokenizer_json);
        result.special_tokens_ = parse_special_tokens(tokenizer_json);
        result.special_token_regex_ = make_special_token_regex(result.special_tokens_);

        return result;
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
                ids.append_range(tokenize_text(piece));
            }
        }

        return ids;
    }

    TokenList Tokenizer::tokenize_text(const std::string& text) const {
        const auto normalized = normalize_nfc(text);
        const auto pieces = pre_tokenize(normalized);

        TokenList ids;
        for (const auto& piece : pieces) {
            const auto symbols = merge_bpe(byte_level_encode(piece));
            for (const auto& symbol : symbols) {
                ids.push_back(vocab_.word_to_id.at(symbol));
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

    std::vector<std::string> Tokenizer::pre_tokenize(const std::string& prompt) const {
        auto begin = boost::make_u32regex_token_iterator(prompt, pretokenizer_regex_);
        auto end = decltype(begin){}; // default constructed iterator type represents end
        return {begin, end};
    }

    std::vector<std::string> Tokenizer::merge_bpe(std::vector<std::string> token_list) const {
        const auto make_merge_pair = [](const std::string& lhs, const std::string& rhs) {
            return MergePair{.left = lhs, .right = rhs};
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
            const auto best = std::ranges::min_element(candidates, [this](const MergePair& lhs, const MergePair& rhs) {
                return merges_.at(lhs) < merges_.at(rhs);
            });

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

} // namespace inference
