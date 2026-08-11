#include <tokenizer/tokenizer.hpp>

#include "util.hpp"

static inference::Tokenizer get_tokenizer() noexcept {
    static const auto tokenizer = [] {
        const auto resource_path = test::resource_path();
        return inference::Tokenizer::load_tokenizer(resource_path / "tokenizer.json", resource_path / "tokenizer_config.json");
    }();

    return tokenizer;
}

TEST(Tokenizer, TokenizesAndDecodesBasicText) {
    const auto tokenizer = get_tokenizer();

    constexpr auto prompt = "Hello world";

    const auto tokens = tokenizer.tokenize(prompt);
    EXPECT_THAT(tokens, testing::ElementsAre(9707, 1879));

    const auto decode = tokenizer.decode(tokens);
    ASSERT_EQ(prompt, decode);
}

TEST(Tokenizer, UnicodeText) {
    const auto tokenizer = get_tokenizer();

    constexpr auto prompt = "曹尼玛💀";

    const auto tokens = tokenizer.tokenize(prompt);
    EXPECT_THAT(tokens, testing::ElementsAre(102263, 99685, 101382, 146571));

    const auto decode = tokenizer.decode(tokens);
    ASSERT_EQ(prompt, decode);
}

TEST(Tokenizer, BasicSpeicalToken) {
    const auto tokenizer = get_tokenizer();

    constexpr auto prompt = "<|endoftext|><tool_response></tool_response>";

    const auto tokens = tokenizer.tokenize(prompt);
    EXPECT_THAT(tokens, testing::ElementsAre(151643, 151665, 151666));

    const auto decode = tokenizer.decode(tokens);
    ASSERT_EQ(prompt, decode);
}
