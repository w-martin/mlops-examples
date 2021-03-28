#include <iostream>
#include <domain/data_generation/mandelbrot_generator_avx.h>
#include "gtest/gtest.h"
#include <chrono>



using namespace std;
namespace {
    TEST(MandelbrotGeneratorAVXTest, GetSmall) {
        auto m = AVX::MandelbrotGenerator();
        auto timeBefore = chrono::high_resolution_clock::now();
        auto result = m.get(8);
        std::stringstream timeStream;
        timeStream << std::fixed << std::setprecision(5) << chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - timeBefore).count() / 1e9;
        cout << "avx mandelbrot rows=3 max=25 took " << timeStream.str() << "s" << endl;
        cout << result << endl;

        timeStream.str(std::string());
        timeBefore = chrono::high_resolution_clock::now();
        m.withMax(255)->get(5000);
        timeStream << std::fixed << std::setprecision(5) << chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - timeBefore).count() / 1e9;
        cout << "avx mandelbrot rows=5000 max=25 took " << timeStream.str() << "s" << endl;


        timeStream.str(std::string());
        timeBefore = chrono::high_resolution_clock::now();
        m.withMax(255)->get(500000);
        timeStream << std::fixed << std::setprecision(5) << chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - timeBefore).count() / 1e9;
        cout << "avx mandelbrot rows=500000 max=255 took " << timeStream.str() << "s" << endl;

        timeStream.str(std::string());
        timeBefore = chrono::high_resolution_clock::now();
        m.withMax(255)->get(5000000);
        timeStream << std::fixed << std::setprecision(5) << chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - timeBefore).count() / 1e9;
        cout << "avx mandelbrot rows=5000000 max=255 took " << timeStream.str() << "s" << endl;
    }
}