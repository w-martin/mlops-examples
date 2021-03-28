//
// Created by will on 26/03/2021.
//

#ifndef MLOPS_EXAMPLES_AVX_MATH_H
#define MLOPS_EXAMPLES_AVX_MATH_H

#include <memory>
#include "immintrin.h"

namespace AVXMATH {

    std::unique_ptr<__m256i> getMask(unsigned short const & length);
    std::unique_ptr<__m128i> getSmallMask(unsigned short const & length);

    void compare(int const *data, unsigned short const &value, long const &size, int *out);

    int sum(int const *data, long const &size);
    void complexSquare(float const * real, float const * imag,
                       unsigned long const &size,
                       float * outReal, float * outImag);
    void add(float const * a, float const * b, unsigned long const & size,
             float * out);
    void mult(float const * a, float const * b, unsigned long const & size,
             float * out);
    void greaterThanScalar(float const * data, const int & threshold, unsigned long const & size,
             int * out);
    void logicalAnd(int const * a, int const * b,
                    unsigned long const & size, int * out);
    void multIntScalar(int const * a, int const & b,
                    unsigned long const & size, int * out);
    void addInt(int const * a, int const * b,
                unsigned long const & size, int * out);
}


#endif //MLOPS_EXAMPLES_AVX_MATH_H
