#include <gtest/gtest.h>
#include "__preprocessor__.h"

TEST(SampleTest, Addition)
{
    EXPECT_EQ(my_sum(1, 1), 2);
}



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
