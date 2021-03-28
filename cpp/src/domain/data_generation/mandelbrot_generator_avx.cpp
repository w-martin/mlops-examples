#include <thread>
#include <cmath>
#include "mandelbrot_generator_avx.h"
#include <iostream>
#include "domain/avx/avx_math.h"

ArrayXi AVX::MandelbrotGenerator::computeMandelbrot(
        const ArrayXf & real, const ArrayXf & imag, const unsigned short &max, const unsigned short &threshold) {
    int size = real.size();
    ArrayXi result = ArrayXi::Zero(size), tmpI(size);
    ArrayXi mask(size);


    ArrayXf zReal(size), zImag(size), tmpF(size);
    zReal << real;
    zImag << imag;

    unsigned short i = 1;
    while (i < max) {
        AVXMATH::compare(result.data(), 0, size, mask.data());
        int sum = AVXMATH::sum(mask.data(), size);
        if (sum == 0) {
            break;
        }
        AVXMATH::complexSquare(zReal.data(), zImag.data(), size,
                               zReal.data(), zImag.data());
        AVXMATH::add(zReal.data(), real.data(), size, zReal.data());
        AVXMATH::add(zImag.data(), imag.data(), size, zImag.data());

        AVXMATH::mult(zReal.data(), zImag.data(), size, tmpF.data());
        AVXMATH::greaterThanScalar(tmpF.data(), threshold, size, tmpI.data());
        AVXMATH::logicalAnd(mask.data(), tmpI.data(), size, mask.data());

        AVXMATH::multIntScalar(mask.data(), -1, size, mask.data());
        AVXMATH::multIntScalar(mask.data(), i, size, mask.data());
        AVXMATH::addInt(result.data(), mask.data(), size, result.data());
        i++;
    }
    AVXMATH::compare(result.data(), 0, size, mask.data());
    AVXMATH::multIntScalar(mask.data(), -1, size, mask.data());
    AVXMATH::multIntScalar(mask.data(), max, size, mask.data());

    AVXMATH::addInt(result.data(), mask.data(), size, result.data());
    return result;
}

void computeMandelbrotAVX(
        const float *cReal, const float *cImag,
        const unsigned long &size, float *out,
        unsigned short max, unsigned short threshold) {

    unsigned short bufferSize = 8;
    unsigned long simdSize = bufferSize * (size / bufferSize);
    unsigned long offset = 0;
    int sum;
    __m256 cr, ci, zr, zi, floatBufferA, floatBufferB;
    __m256 resultMask, logicalBuffer;
    __m128 smallBufferA, smallBufferB;
    const __m256 zeroBuffer = _mm256_setzero_ps();
    const __m256 thresholdBuffer = _mm256_set1_ps(threshold);
    const __m256 twoBuffer = _mm256_set1_ps(2);
    const __m256 oneBuffer = _mm256_set1_ps(1);
    std::unique_ptr<__m256i> mask = AVXMATH::getMask(8);

    for (; offset < size; offset+=bufferSize) {
        if (offset == simdSize) {
            mask = AVXMATH::getMask(size - offset);
        }
        zr = cr = _mm256_maskload_ps((const float *) (cReal + offset), *mask);
        zi = ci = _mm256_maskload_ps((const float *) (cImag + offset), *mask);
        resultMask = zeroBuffer;
        for (unsigned short i = 1; i <= max; i++) {
            // check unset
            logicalBuffer = _mm256_cmp_ps(resultMask, zeroBuffer, _CMP_EQ_OS);
            // sum unset
            smallBufferA = _mm256_extractf128_ps(logicalBuffer, 0);
            smallBufferB = _mm256_extractf128_ps(logicalBuffer, 1);
            smallBufferA = _mm_add_ps(smallBufferA, smallBufferB);
            sum = 0;
            for (unsigned short sumCounter = 0; sumCounter < 4; sumCounter++) {
                sum += _mm_extract_ps(smallBufferA, sumCounter);
            }
            // break if all set
            if (sum == 0) {
                break;
            }
            // z = z^2 + c
            floatBufferA = _mm256_mul_ps(zi, zi);
            floatBufferA = _mm256_fmsub_ps(zr, zr, floatBufferA);
            floatBufferB = _mm256_mul_ps(zr, zi);
            floatBufferB = _mm256_mul_ps(floatBufferB, twoBuffer);
            zr = _mm256_add_ps(floatBufferA, cr);
            zi = _mm256_add_ps(floatBufferB, ci);
            // check inner product greater than a threshold
            if (i < max) {
                floatBufferA = _mm256_mul_ps(zr, zi);
                floatBufferA = _mm256_cmp_ps(floatBufferA, thresholdBuffer, _CMP_GT_OQ);
                logicalBuffer = _mm256_and_ps(logicalBuffer, floatBufferA);
            }
            logicalBuffer = _mm256_and_ps(logicalBuffer, oneBuffer);
            // add to result
            floatBufferB = _mm256_set1_ps(i);
            floatBufferA = _mm256_mul_ps(logicalBuffer, floatBufferB);
            resultMask = _mm256_add_ps(resultMask, floatBufferA);
        }
        // set result
        _mm256_maskstore_ps((float *) (out + offset), *mask, resultMask);
    }
}

ArrayXXf AVX::MandelbrotGenerator::get(int nRows) {
    int rootN = static_cast<int>(ceil(sqrt(nRows)));
    ArrayXf real = ArrayXf::LinSpaced(rootN, R_MIN, R_MAX)
            .replicate(rootN, 1);
    ArrayXf imag = ArrayXf::LinSpaced(rootN, I_MIN, I_MAX)
            .replicate(rootN, 1);
    std::sort(real.begin(), real.end(), std::less<>());
    unsigned long size = real.size();
    ArrayXf y(size);
//    unsigned short nThreads = std::thread::hardware_concurrency();
//    std::cout << "Running with " << nThreads << " threads" << std::endl;
//    unsigned long partitionSize = size / nThreads;
//    auto *threads = new std::thread[nThreads];
//    for (unsigned short i = 0; i < nThreads; i++) {
//        unsigned long offset = partitionSize * (unsigned long) i;
//        unsigned long thisPartitionSize = i == (nThreads - 1) ? (size - ((nThreads - 1) * partitionSize)) : partitionSize;
//        threads[i] = std::thread(&computeMandelbrotAVX, real.data() + offset, imag.data() + offset, thisPartitionSize, y.data() + offset, max, threshold);
//    }
//    std::cout << "Started " << nThreads << " threads" << std::endl;
//    for (unsigned int i = 0; i < nThreads; i++) {
//        threads[i].join();
//    }
//    std::cout << "Finished " << nThreads << " threads" << std::endl;
    computeMandelbrotAVX(real.data(), imag.data(), size, y.data(), max, threshold);
    ArrayXXf result(size, 3);
    result.col(0) << real;
    result.col(1) << imag;
    result.col(2) << y;
    return result;
}
