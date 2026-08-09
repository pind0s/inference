#include <tokenizer/tokenizer.hpp>

#include "util.hpp"

static inference::Tokenizer get_tokenizer() noexcept {
    auto resource_path = test::resource_path();
    return inference::Tokenizer::load_tokenizer(resource_path / "tokenizer.json", resource_path / "tokenizer_config.json");
}

const static inference::Tokenizer tokenizer = get_tokenizer();

TEST(Tokenizer, TokenizesAndDecodesBasicText) {
    constexpr auto prompt = "Hello world";

    const auto tokens = tokenizer.tokenize(prompt);
    EXPECT_THAT(tokens, testing::ElementsAre(9707, 1879));

    const auto decode = tokenizer.decode(tokens);
    ASSERT_EQ(prompt, decode);
}

TEST(Tokenizer, UnicodeText) {
    constexpr auto prompt = "曹尼玛💀";

    const auto tokens = tokenizer.tokenize(prompt);
    EXPECT_THAT(tokens, testing::ElementsAre(102263, 99685, 101382, 146571));

    const auto decode = tokenizer.decode(tokens);
    ASSERT_EQ(prompt, decode);
}

TEST(Tokenizer, BasicSpeicalToken) {
    constexpr auto prompt = "<|endoftext|><tool_response></tool_response>";

    const auto tokens = tokenizer.tokenize(prompt);
    EXPECT_THAT(tokens, testing::ElementsAre(151643, 151665, 151666));

    const auto decode = tokenizer.decode(tokens);
    ASSERT_EQ(prompt, decode);
}
