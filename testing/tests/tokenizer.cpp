#include "tokenizer/tokenizer.hpp"
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <string>

TEST_CASE("tokenizer matches known tokens and decodes the original text", "[tokenizer]") {
    using namespace inference::tokenizer;

    static const Tokenizer tokenizer{ std::filesystem::path{ TEST_RESOURCE_DIR } };
    std::string prompt;
    TokenList expected;

    SECTION("basic text") {
        prompt = "Hello world";
        expected = { 9707, 1879 };
    }

    SECTION("Unicode text") {
        prompt = "曹尼玛💀";
        expected = { 102263, 99685, 101382, 146571 };
    }

    SECTION("special tokens") {
        prompt = "<|endoftext|><tool_response></tool_response>";
        expected = { 151643, 151665, 151666 };
    }

    const auto tokens = tokenizer.tokenize(prompt);
    CHECK(tokens == expected);
    CHECK(tokenizer.decode(tokens) == prompt);
}
