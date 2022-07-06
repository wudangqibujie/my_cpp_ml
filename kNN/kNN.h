//
// Created by Administrator on 2022/7/6 0006.
//

#ifndef UNTITLED_KNN_H
#define UNTITLED_KNN_H

#include <vector>

namespace MLPP{
    class kNN{

    public:
        kNN(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, int k);
        std::vector<double> modelSetTest(std::vector<std::vector<double>> X);
        int modelTest(std::vector<double> x);
        double score();

    private:

        // Private Model Functions
        std::vector<double> nearestNeighbors(std::vector<double> x);
        int determineClass(std::vector<double> knn);

        // Model Inputs and Parameters
        std::vector<std::vector<double>> inputSet;
        std::vector<double> outputSet;
        int k;

    };
}
#endif //UNTITLED_KNN_H
