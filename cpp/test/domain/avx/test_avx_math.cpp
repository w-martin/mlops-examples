#include <iostream>
#include <domain/avx/avx_math.h>
#include "gtest/gtest.h"
#include <chrono>
#include <Eigen/Dense>

using namespace Eigen;
namespace {
    TEST(MandelbrotGeneratorAVXTest, AvxCompare) {
        // arrange
        int size = 18;
        ArrayXi a(size);
        a << 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1;
        ArrayXi b = 1 - a;
        Array<int, -1, 1> c(size);
        std::cout << "A: " << a.transpose() << std::endl;
        // act
        AVXMATH::compare(a.data(), 0, size, c.data());
        // assert
        Array<bool, -1, 1> bb = b.cast<bool>();
        Array<bool, -1, 1> cb = c.cast<bool>();
        std::cout << "B: " << bb.transpose() << std::endl;
        std::cout << "C: " << cb.transpose() << std::endl;
        for (int i = 0; i < size; i++) {
            ASSERT_EQ(bb(i), cb(i));
        }
    }

    TEST(MandelbrotGeneratorAVXTest, ShouldSum32) {
        // arrange
        ArrayXi data = ArrayXi::LinSpaced(100, 0, 100);
        int expected = data.sum();
        // act
        int actual = AVXMATH::sum(data.data(), data.size());
        // assert
        ASSERT_EQ(expected, actual);
    }

    TEST(MandelbrotGeneratorAVXTest, ShouldSum16) {
        // arrange
        ArrayXi data = ArrayXi::LinSpaced(7, 0, 100);
        int expected = data.sum();
        // act
        int actual = AVXMATH::sum(data.data(), data.size());
        // assert
        ASSERT_EQ(expected, actual);
    }

    TEST(MandelbrotGeneratorAVXTest, ShouldSumSingleBuffer) {
        // arrange
        ArrayXi data = ArrayXi::LinSpaced(3, 0, 100);
        int expected = data.sum();
        // act
        int actual = AVXMATH::sum(data.data(), data.size());
        // assert
        ASSERT_EQ(expected, actual);
    }

    TEST(MandelbrotGeneratorAVXTest, AvxComplexSquare) {
        // arrange
        int n = 100;
        ArrayXcf data = ArrayXcf::Zero(n, 1);
        ArrayXf real = ArrayXf::LinSpaced(n, -1.1, 1.1)
                .replicate(n, 1);
        ArrayXf imag = ArrayXf::LinSpaced(n, -1.1, 1.1)
                .replicate(n, 1);
        std::sort(real.begin(), real.end(), std::less<>());
        data += real.segment(0, n);
        data.imag() += imag.segment(0, n);
        ArrayXcf expected = data.square();
        ArrayXf actualReal(n);
        ArrayXf actualImag(n);
        // act
        AVXMATH::complexSquare(real.data(), imag.data(),
                               data.size(),
                               actualReal.data(), actualImag.data());
        // assert
        ASSERT_TRUE(expected.real().cwiseEqual(actualReal).all());
        ASSERT_TRUE(expected.imag().cwiseEqual(actualImag).all());
    }

    TEST(MandelbrotGeneratorAVXTest, AvxAdd) {
        // arrange
        int n = 100;
        ArrayXf real = ArrayXf::Random(n);
        ArrayXf imag = ArrayXf::Random(n);
        ArrayXf expected = real + imag;
        ArrayXf actual(n);
        // act
        AVXMATH::add(real.data(), imag.data(), n, actual.data());
        // assert
        ASSERT_TRUE(expected.cwiseEqual(actual).all());
    }

    TEST(MandelbrotGeneratorAVXTest, AvxMult) {
        // arrange
        int n = 100;
        ArrayXf real = ArrayXf::Random(n);
        ArrayXf imag = ArrayXf::Random(n);
        ArrayXf expected = real * imag;
        ArrayXf actual(n);
        // act
        AVXMATH::mult(real.data(), imag.data(), n, actual.data());
        // assert
        ASSERT_TRUE(expected.cwiseEqual(actual).all());
    }

    TEST(MandelbrotGeneratorAVXTest, GreaterThanScalar) {
        // arrange
        int n = 100;
        ArrayXf data = ArrayXf::Random(n);
        ArrayX<bool> expected = data > 0;
        ArrayXi actual(n);
        // act
        AVXMATH::greaterThanScalar(data.data(), 0, n, actual.data());
        // assert
        ASSERT_TRUE(expected.cwiseEqual(actual.cast<bool>()).all());
    }

    TEST(MandelbrotGeneratorAVXTest, LogicalAnd) {
        // arrange
        int n = 100;
        ArrayX<bool> a = ArrayXf::Random(n) > 0;
        ArrayX<bool> b = ArrayXf::Random(n) > 0;
        ArrayX<bool> expected = a && b;
        ArrayXi actual(n), ai = a.cast<int>() * -1, bi = b.cast<int>() * -1;
        // act
        AVXMATH::logicalAnd(ai.data(), bi.data(), n, actual.data());
        // assert
        ASSERT_TRUE(expected.cwiseEqual(actual.cast<bool>()).all());
    }

    TEST(MandelbrotGeneratorAVXTest, MultIntScalar) {
        // arrange
        int n = 100;
        ArrayX<int> data = (ArrayXf::Random(n) > 0).cast<int>();
        ArrayX<int> expected = data * 15;
        ArrayXi actual(n);
        // act
        AVXMATH::multIntScalar(data.data(), 15, n, actual.data());
        // assert
        ASSERT_TRUE(expected.cwiseEqual(actual).all());
    }

    TEST(MandelbrotGeneratorAVXTest, ShouldAddIntArrays) {
        // arrange
        int n = 100;
        ArrayX<int> a = (ArrayXf::Random(n) * 100).cast<int>();
        ArrayX<int> b = (ArrayXf::Random(n) * 100).cast<int>();
        ArrayX<int> expected = a + b;
        ArrayXi actual(n);
        // act
        AVXMATH::addInt(a.data(), b.data(), n, actual.data());
        // assert
        ASSERT_TRUE(expected.cwiseEqual(actual).all());
    }

}