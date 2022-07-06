//
// Created by Administrator on 2022/7/6 0006.
//

#ifndef UNTITLED_SVC_H
#define UNTITLED_SVC_H

#include <vector>
#include <string>

namespace MLPP {

    class SVC{

    public:
        SVC(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, double C);
        std::vector<double> modelSetTest(std::vector<std::vector<double>> X);
        double modelTest(std::vector<double> x);
        void gradientDescent(double learning_rate, int max_epoch, bool UI = 1);
        void SGD(double learning_rate, int max_epoch, bool UI = 1);
        void MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI = 1);
        double score();
        void save(std::string fileName);
    private:

        double Cost(std::vector <double> y_hat, std::vector<double> y, std::vector<double> weights, double C);

        std::vector<double> Evaluate(std::vector<std::vector<double>> X);
        std::vector<double> propagate(std::vector<std::vector<double>> X);
        double Evaluate(std::vector<double> x);
        double propagate(std::vector<double> x);
        void forwardPass();

        std::vector<std::vector<double>> inputSet;
        std::vector<double> outputSet;
        std::vector<double> z;
        std::vector<double> y_hat;
        std::vector<double> weights;
        double bias;

        double C;
        int n;
        int k;

        // UI Portion
        void UI(int epoch, double cost_prev);
    };
}
#endif //UNTITLED_SVC_H
