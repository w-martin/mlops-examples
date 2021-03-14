#include <iostream>
#include <domain/data_generation/mandelbrot_generator.h>
#include "gtest/gtest.h"


namespace {


    TEST(DataGeneratorTest, WithMax) {
        auto m = MandelbrotGenerator();
        auto result = m.withMax(42);
        EXPECT_EQ(&m, result);
    }
}