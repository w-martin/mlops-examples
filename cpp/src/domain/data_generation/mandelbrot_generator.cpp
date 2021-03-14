
#include <cmath>
#include "mandelbrot_generator.h"
#include <iostream>

ArrayXi MandelbrotGenerator::computeMandelbrot(ArrayXcf c) {
    ArrayXi result = ArrayXi::Constant(c.size(), max);
    ArrayXcf z(c.size());
    z << c;
    short i = 1;

    while (i < max) {
        auto unset = result == max;
        if (unset.sum() == 0) {
            break;
        }
        z = z.square() + c;
        auto mask = ((z.real() * z.imag()) > threshold) && unset;
        result = mask.select(i, result);
        i++;
    }
    return result;
}

ArrayXXf MandelbrotGenerator::get(int nRows) {
    int rootN = static_cast<int>(ceil(sqrt(nRows)));
    ArrayXcf data = ArrayXcf::Zero(nRows, 1);
    ArrayXf real = ArrayXf::LinSpaced(rootN, R_MIN, R_MAX)
            .replicate(rootN, 1);
    ArrayXf imag = ArrayXf::LinSpaced(rootN, I_MIN, I_MAX)
            .replicate(rootN, 1);
    std::sort(real.begin(), real.end(), std::less<>());
    data += real.segment(0, nRows);
    data.imag() += imag.segment(0, nRows);
    ArrayXi y = computeMandelbrot(data);
    ArrayXXf result(nRows, 3);
    result.col(0) << data.real();
    result.col(1) << data.imag();
    result.col(2) << y.cast<float>();
    return result;
}
