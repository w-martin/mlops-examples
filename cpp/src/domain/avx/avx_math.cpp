
#include "avx_math.h"

std::unique_ptr<__m256i> AVXMATH::getMask(unsigned short const &length) {
    __m256i mask;
    switch (length) {
        case 0:
            mask = _mm256_setr_epi32((int) 0x00000000, (int) 0x00000000, (int) 0x00000000, (int) 0x00000000,
                                     (int) 0x00000000, (int) 0x00000000, (int) 0x00000000, (int) 0x00000000);
            break;

        case 1:
            mask = _mm256_setr_epi32((int) 0xf0000000, (int) 0x00000000, (int) 0x00000000, (int) 0x00000000,
                                     (int) 0x00000000, (int) 0x00000000, (int) 0x00000000, (int) 0x00000000);
            break;
        case 2:
            mask = _mm256_setr_epi32((int) 0xf0000000, (int) 0xf0000000, (int) 0x00000000, (int) 0x00000000,
                                     (int) 0x00000000, (int) 0x00000000, (int) 0x00000000, (int) 0x00000000);
            break;
        case 3:
            mask = _mm256_setr_epi32((int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000, (int) 0x00000000,
                                     (int) 0x00000000, (int) 0x00000000, (int) 0x00000000, (int) 0x00000000);
            break;
        case 4:
            mask = _mm256_setr_epi32((int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000,
                                     (int) 0x00000000, (int) 0x00000000, (int) 0x00000000, (int) 0x00000000);
            break;
        case 5:
            mask = _mm256_setr_epi32((int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000,
                                     (int) 0xf0000000, (int) 0x00000000, (int) 0x00000000, (int) 0x00000000);
            break;
        case 6:
            mask = _mm256_setr_epi32((int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000,
                                     (int) 0xf0000000, (int) 0xf0000000, (int) 0x00000000, (int) 0x00000000);
            break;
        case 7:
            mask = _mm256_setr_epi32((int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000,
                                     (int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000, (int) 0x00000000);
            break;
        case 8:
            mask = _mm256_setr_epi32((int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000,
                                     (int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000);
    }
    return std::make_unique<__m256i>(mask);
};

std::unique_ptr<__m128i> AVXMATH::getSmallMask(unsigned short const &length) {
    __m128i mask;
    switch (length) {
        case 0:
            mask = _mm_setr_epi32((int) 0x00000000, (int) 0x00000000, (int) 0x00000000, (int) 0x00000000);
            break;
        case 1:
            mask = _mm_setr_epi32((int) 0xf0000000, (int) 0x00000000, (int) 0x00000000, (int) 0x00000000);
            break;
        case 2:
            mask = _mm_setr_epi32((int) 0xf0000000, (int) 0xf0000000, (int) 0x00000000, (int) 0x00000000);
            break;
        case 3:
            mask = _mm_setr_epi32((int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000, (int) 0x00000000);
            break;
        case 4:
            mask = _mm_setr_epi32((int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000, (int) 0xf0000000);
    }
    return std::make_unique<__m128i>(mask);
};


void AVXMATH::compare(int const *data, unsigned short const &value, long const &size, int *out) {
    __m256i valueBuffer = _mm256_setzero_si256();
    unsigned short bufferSize = 8;
    unsigned long simdSize = bufferSize * (size / bufferSize);
    int offset = 0;
    for (; offset < simdSize; offset += bufferSize) {
        __m256i buffer = _mm256_load_si256((__m256i const *) (data + offset));
        __m256i batchResult = _mm256_cmpeq_epi32(valueBuffer, buffer);
        _mm256_store_si256((__m256i * )(out + offset), batchResult);
    }
    std::unique_ptr<__m256i> mask = AVXMATH::getMask(size - simdSize);
    __m256i buffer = _mm256_maskload_epi32((const int *) (data + simdSize), *mask);
    __m256i batchResult = _mm256_cmpeq_epi32(valueBuffer, buffer);
    _mm256_maskstore_epi32((int *) (out + offset), *mask, batchResult);
}


int AVXMATH::sum(const int *data, const long &size) {
    int sum = 0;
    unsigned short bufferSize = 8, smallBufferSize = 4;
    unsigned long simdSize = bufferSize * (size / bufferSize);
    __m128i smallBuffer;
    if (size > bufferSize) {
        __m256i buffer;
        __m256i resultBuffer = _mm256_load_si256((__m256i const *) data);
        unsigned long offset;
        for (offset = bufferSize; offset < simdSize; offset += bufferSize) {
            buffer = _mm256_load_si256((__m256i const *) (data + offset));
            resultBuffer = _mm256_add_epi32(resultBuffer, buffer);
        }
        unsigned short remaining = size - offset;
        if (remaining > 0) {
            std::unique_ptr<__m256i> mask = AVXMATH::getMask(remaining);
            buffer = _mm256_maskload_epi32((const int *) (data + simdSize), *mask);
            resultBuffer = _mm256_add_epi32(resultBuffer, buffer);
        }
        smallBuffer = _mm256_extracti128_si256(resultBuffer, 0);
        smallBuffer = _mm_add_epi32(smallBuffer, _mm256_extracti128_si256(resultBuffer, 1));
    } else if (size > smallBufferSize) {
        std::unique_ptr<__m128i> mask = AVXMATH::getSmallMask(size - smallBufferSize);
        smallBuffer = _mm_maskload_epi32((int const *) (data + smallBufferSize), *mask);
        smallBuffer = _mm_add_epi32(smallBuffer, _mm_load_si128((__m128i const *) data));
    } else {
        std::unique_ptr<__m128i> mask = AVXMATH::getSmallMask(size);
        smallBuffer = _mm_maskload_epi32((int const *) data, *mask);
    }
    for (int i = 0; i < smallBufferSize; i++) {
        sum += _mm_extract_epi32(smallBuffer, i);
    }
    return sum;
}


void AVXMATH::complexSquare(const float *real, const float *imag,
                            const unsigned long &size,
                            float *outReal,            float *outImag) {
    unsigned short bufferSize = 8;
    unsigned long simdSize = bufferSize * (size / bufferSize);
    unsigned long offset = 0;
    __m256 a, b, c, d;
    __m256 two = _mm256_set1_ps(2);
    for (; offset < simdSize; offset+=bufferSize) {
        a = _mm256_load_ps((const float *) (real + offset));
        b = _mm256_load_ps((const float *) (imag + offset));
        c = _mm256_mul_ps(a, a);
        d = _mm256_mul_ps(b, b);
        c = _mm256_sub_ps(c, d);
        _mm256_store_ps((float * )(outReal + offset), c);
        c = _mm256_mul_ps(a, b);
        c = _mm256_mul_ps(c, two);
        _mm256_store_ps((float * )(outImag + offset), c);
    }
    unsigned short remaining = size - offset;
    if (remaining > 0) {
        std::unique_ptr<__m256i> mask = AVXMATH::getMask(remaining);
        a = _mm256_maskload_ps((const float *) (real + offset), *mask);
        b = _mm256_maskload_ps((const float *) (imag + offset), *mask);
        c = _mm256_mul_ps(a, a);
        d = _mm256_mul_ps(b, b);
        c = _mm256_sub_ps(c, d);
        _mm256_maskstore_ps((float * )(outReal + offset), *mask, c);
        c = _mm256_mul_ps(a, b);
        c = _mm256_mul_ps(c, two);
        _mm256_maskstore_ps((float * )(outImag + offset), *mask, c);
    }
}

void AVXMATH::add(const float *a, const float *b, const unsigned long &size, float *out) {
    unsigned short bufferSize = 8;
    unsigned long simdSize = bufferSize * (size / bufferSize);
    unsigned long offset = 0;
    __m256 c, d;
    for (; offset < simdSize; offset+=bufferSize) {
        c = _mm256_load_ps((const float *) (a + offset));
        d = _mm256_load_ps((const float *) (b + offset));
        c = _mm256_add_ps(c, d);
        _mm256_store_ps((float * )(out + offset), c);
    }
    unsigned short remaining = size - offset;
    if (remaining > 0) {
        std::unique_ptr<__m256i> mask = AVXMATH::getMask(remaining);
        c = _mm256_maskload_ps((const float *) (a + offset), *mask);
        d = _mm256_maskload_ps((const float *) (b + offset), *mask);
        c = _mm256_add_ps(c, d);
        _mm256_maskstore_ps((float * )(out + offset), *mask, c);
    }
}


void AVXMATH::mult(const float *a, const float *b, const unsigned long &size, float *out) {
    unsigned short bufferSize = 8;
    unsigned long simdSize = bufferSize * (size / bufferSize);
    unsigned long offset = 0;
    __m256 c, d;
    for (; offset < simdSize; offset+=bufferSize) {
        c = _mm256_load_ps((const float *) (a + offset));
        d = _mm256_load_ps((const float *) (b + offset));
        c = _mm256_mul_ps(c, d);
        _mm256_store_ps((float * )(out + offset), c);
    }
    unsigned short remaining = size - offset;
    if (remaining > 0) {
        std::unique_ptr<__m256i> mask = AVXMATH::getMask(remaining);
        c = _mm256_maskload_ps((const float *) (a + offset), *mask);
        d = _mm256_maskload_ps((const float *) (b + offset), *mask);
        c = _mm256_mul_ps(c, d);
        _mm256_maskstore_ps((float * )(out + offset), *mask, c);
    }
}


void AVXMATH::greaterThanScalar(const float *data, const int &threshold, const unsigned long &size, int *out) {
    unsigned short bufferSize = 8;
    unsigned long simdSize = bufferSize * (size / bufferSize);
    unsigned long offset = 0;
    __m256i valueBuffer = _mm256_set1_epi32(threshold);
    __m256 buffer;
    __m256i resultBuffer, intBuffer;
    for (; offset < simdSize; offset+=bufferSize) {
        buffer = _mm256_load_ps((const float *) (data + offset));
        intBuffer = _mm256_castps_si256(buffer);
        resultBuffer = _mm256_cmpgt_epi32(intBuffer, valueBuffer);
        _mm256_store_si256((__m256i * )(out + offset), resultBuffer);
    }
    unsigned short remaining = size - offset;
    if (remaining > 0) {
        std::unique_ptr<__m256i> mask = AVXMATH::getMask(remaining);
        buffer = _mm256_maskload_ps((const float *) (data + offset), *mask);
        intBuffer = _mm256_castps_si256(buffer);
        resultBuffer = _mm256_cmpgt_epi32(intBuffer, valueBuffer);
        _mm256_maskstore_epi32((int * )(out + offset), *mask, resultBuffer);
    }
}

void AVXMATH::logicalAnd(const int *a, const int *b,
                         const unsigned long &size, int *out) {
    unsigned short bufferSize = 8;
    unsigned long simdSize = bufferSize * (size / bufferSize);
    unsigned long offset = 0;
    __m256i c, d;
    for (; offset < simdSize; offset+=bufferSize) {
        c = _mm256_load_si256((const __m256i *) (a + offset));
        d = _mm256_load_si256((const __m256i *) (b + offset));
        c = _mm256_and_si256(c, d);
        _mm256_store_si256((__m256i * )(out + offset), c);
    }
    unsigned short remaining = size - offset;
    if (remaining > 0) {
        std::unique_ptr<__m256i> mask = AVXMATH::getMask(remaining);
        c = _mm256_maskload_epi32((const int *) (a + offset), *mask);
        d = _mm256_maskload_epi32((const int *) (b + offset), *mask);
        c = _mm256_and_si256(c, d);
        _mm256_maskstore_epi32((int * )(out + offset), *mask, c);
    }
}

void AVXMATH::multIntScalar(const int *a, const int & b,
                         const unsigned long &size, int *out) {
    unsigned short bufferSize = 8;
    unsigned long simdSize = bufferSize * (size / bufferSize);
    unsigned long offset = 0;
    __m256i valueBuffer = _mm256_set1_epi32(b);
    __m256i buffer;
    for (; offset < simdSize; offset+=bufferSize) {
        buffer = _mm256_load_si256((const __m256i *) (a + offset));
        buffer = _mm256_mullo_epi32(buffer, valueBuffer);
        _mm256_store_si256((__m256i * )(out + offset), buffer);
    }
    unsigned short remaining = size - offset;
    if (remaining > 0) {
        std::unique_ptr<__m256i> mask = AVXMATH::getMask(remaining);
        buffer = _mm256_maskload_epi32((const int *) (a + offset), *mask);
        buffer = _mm256_mullo_epi32(buffer, valueBuffer);
        _mm256_maskstore_epi32((int * )(out + offset), *mask, buffer);
    }
}

void AVXMATH::addInt(const int *a, const int * b,
                         const unsigned long &size, int *out) {
    unsigned short bufferSize = 8;
    unsigned long simdSize = bufferSize * (size / bufferSize);
    unsigned long offset = 0;
    __m256i c, d;
    for (; offset < simdSize; offset+=bufferSize) {
        c = _mm256_load_si256((const __m256i *) (a + offset));
        d = _mm256_load_si256((const __m256i *) (b + offset));
        c = _mm256_add_epi32(c, d);
        _mm256_store_si256((__m256i * )(out + offset), c);
    }
    unsigned short remaining = size - offset;
    if (remaining > 0) {
        std::unique_ptr<__m256i> mask = AVXMATH::getMask(remaining);
        c = _mm256_maskload_epi32((const int *) (a + offset), *mask);
        d = _mm256_maskload_epi32((const int *) (b + offset), *mask);
        c = _mm256_add_epi32(c, d);
        _mm256_maskstore_epi32((int * )(out + offset), *mask, c);
    }
}
