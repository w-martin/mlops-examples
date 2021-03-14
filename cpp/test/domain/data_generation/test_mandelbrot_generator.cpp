#include <iostream>
#include <domain/data_generation/mandelbrot_generator.h>
#include "gtest/gtest.h"
#include <chrono>

using namespace std;
namespace {


    TEST(MandelbrotGeneratorTest, GetSmall) {
        auto m = MandelbrotGenerator();
        auto timeBefore = chrono::high_resolution_clock::now();
        auto result = m.get(3);
        std::stringstream timeStream;
        timeStream << std::fixed << std::setprecision(5) << chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - timeBefore).count() / 1e9;
        cout << "mandelbrot rows=3 max=25 took " << timeStream.str() << "s" << endl;
        cout << result << endl;

        timeStream.str(std::string());
        timeBefore = chrono::high_resolution_clock::now();
        m.get(5000);
        timeStream << std::fixed << std::setprecision(5) << chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - timeBefore).count() / 1e9;
        cout << "mandelbrot rows=5000 max=25 took " << timeStream.str() << "s" << endl;


        timeStream.str(std::string());
        timeBefore = chrono::high_resolution_clock::now();
        m.withMax(255)->get(5000);
        timeStream << std::fixed << std::setprecision(5) << chrono::duration_cast<chrono::nanoseconds>(
                chrono::high_resolution_clock::now() - timeBefore).count() / 1e9;
        cout << "mandelbrot rows=5000 max=255 took " << timeStream.str() << "s" << endl;
    }
}