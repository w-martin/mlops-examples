//
// Created by will on 13/03/2021.
//

#ifndef MLOPSEXAMPLES_MANDELBROT_GENERATOR_H
#define MLOPSEXAMPLES_MANDELBROT_GENERATOR_H


#include "data_generator.h"

class MandelbrotGenerator : public DataGenerator {
public:
    ArrayXXf get(int nRows) override;

protected:
    ArrayXi computeMandelbrot(const ArrayXcf &c, const unsigned short &max, const unsigned short &threshold);
};



#endif //MLOPSEXAMPLES_MANDELBROT_GENERATOR_H
