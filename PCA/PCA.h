//
// Created by Administrator on 2022/7/6 0006.
//

#ifndef UNTITLED_PCA_H
#define UNTITLED_PCA_H

#include <vector>

namespace MLPP{
    class PCA{

    public:
        PCA(std::vector<std::vector<double>> inputSet, int k);
        std::vector<std::vector<double>> principalComponents();
        double score();
    private:
        std::vector<std::vector<double>> inputSet;
        std::vector<std::vector<double>> X_normalized;
        std::vector<std::vector<double>> U_reduce;
        std::vector<std::vector<double>> Z;
        int k;
    };
}
#endif //UNTITLED_PCA_H
