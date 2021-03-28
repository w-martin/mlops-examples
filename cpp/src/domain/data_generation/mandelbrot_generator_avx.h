#ifndef MLOPSEXAMPLES_MANDELBROT_GENERATOR_AVX_H
#define MLOPSEXAMPLES_MANDELBROT_GENERATOR_AVX_H

#include <vector>
#include "data_generator.h"
#include <memory>
#include <unordered_map>

namespace AVX {
    class MandelbrotGenerator : public DataGenerator {
    public:
        ArrayXXf get(int nRows) override;

    protected:
        ArrayXi computeMandelbrot(const ArrayXf &real, const ArrayXf &imag, const unsigned short &max,
                                  const unsigned short &threshold);

//        static void computeMandelbrotAVX(
//                float const *cReal, float const *cImag,
//                unsigned long const &size, float *out,
//                unsigned short max, unsigned short threshold);
    };
}

#endif //MLOPSEXAMPLES_MANDELBROT_GENERATOR_AVX_H
