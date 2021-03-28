
#include <cmath>
#include "mandelbrot_generator.h"
#include <iostream>


ArrayXi MandelbrotGenerator::computeMandelbrot(ArrayXcf const & c, unsigned short const & max, unsigned short const & threshold) {
    ArrayXi result = ArrayXi::Zero(c.size());
    ArrayXcf z(c.size());
    z << c;
    unsigned short i = 1;
    Array<bool, Dynamic, 1> mask(c.size());

    while (i < max) {
        mask = result.cwiseEqual(0);
        if (mask.sum() == 0) {
            break;
        }
        z = z.square() + c;

        mask = mask && ((z.real() * z.imag()) > threshold);
        result += mask.cast<int>() * i;
        i++;
    }
    result += (1 - result.cast<bool>().cast<int>()) * max;
    return result;
}

ArrayXXf MandelbrotGenerator::get(int nRows) {
    int rootN = static_cast<int>(ceil(sqrt(nRows)));
    ArrayXf real = ArrayXf::LinSpaced(rootN, R_MIN, R_MAX)
            .replicate(rootN, 1);
    ArrayXf imag = ArrayXf::LinSpaced(rootN, I_MIN, I_MAX)
            .replicate(rootN, 1);
    std::sort(real.begin(), real.end(), std::less<>());

    long size = real.size();
    ArrayXcf data = ArrayXcf::Zero(size, 1);
    data += real;
    data.imag() += imag;
    ArrayXi const y = computeMandelbrot(data, max, threshold);

    ArrayXXf result(size, 3);
    result.col(0) << data.real();
    result.col(1) << data.imag();
    result.col(2) << y.cast<float>();
    return result;
}
