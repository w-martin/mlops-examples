
#include <domain/data_generation/mandelbrot_generator.h>
#include <search.h>

class DatasetRepository {

//    float[][] get () {
//        float[][] data = new float[][];
//        return data;
//    }
};

int main() {

//    auto datasetRepository = new DatasetRepository();
//    auto data = datasetRepository.get();
//
//    std::cout << data << std::endl;

    auto m = MandelbrotGenerator();
    m.get(50);

    return 0;
}
