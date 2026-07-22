#include <gtest/gtest.h>

#include "example.hpp"

TEST(example, example) {
    EXPECT_EQ(example(), 1);
}

TEST(example, failing_example) {
    EXPECT_EQ(example(), 0);
}