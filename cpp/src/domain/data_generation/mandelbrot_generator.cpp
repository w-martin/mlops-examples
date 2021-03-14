
#include <cmath>
#include "mandelbrot_generator.h"
#include <iostream>


ArrayXi MandelbrotGenerator::computeMandelbrot(ArrayXcf c) {
    ArrayXi result = ArrayXi::Zero(c.size());
    ArrayXcf z(c.size());
    z << c;
    unsigned short i = 1;
    Array<bool, Dynamic, 1> unset(c.size());
    Array<bool, Dynamic, 1> mask(c.size());

    while (i < max) {
        unset = result == 0;
        if (unset.sum() == 0) {
            break;
        }
        z = z.square() + c;
        mask = unset && ((z.real() * z.imag()) > threshold);
        result += mask.cast<int>() * i;
        i++;
    }
    result += (1 - result.cast<bool>().cast<int>()) * max;
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
