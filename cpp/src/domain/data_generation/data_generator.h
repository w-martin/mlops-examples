//
// Created by will on 13/03/2021.
//

#ifndef MLOPSEXAMPLES_DATA_GENERATOR_H
#define MLOPSEXAMPLES_DATA_GENERATOR_H

#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>

using namespace Eigen;


class DataGenerator {
public:
    virtual ArrayXXf get(int n_rows) = 0;

    auto withMax(unsigned short newMax) {
        this->max = newMax;
        return this;
    }

protected:
    unsigned short max = 25;
    unsigned short threshold = 4;
    const float R_MIN = -2;
    const float R_MAX = 0.47;
    const float
            I_MIN = -1.12;
    const float
            I_MAX = 1.12;
};

#endif //MLOPSEXAMPLES_DATA_GENERATOR_H
