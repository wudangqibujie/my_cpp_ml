//
// Created by Administrator on 2022/7/6 0006.
//

#ifndef UNTITLED_DUALSVC_H
#define UNTITLED_DUALSVC_H

#include <vector>
#include <string>

namespace MLPP {

    class DualSVC{

    public:
        DualSVC(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, double C, std::string kernel = "Linear");
        DualSVC(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, double C, std::string kernel, double p, double c);

        std::vector<double> modelSetTest(std::vector<std::vector<double>> X);
        double modelTest(std::vector<double> x);
        void gradientDescent(double learning_rate, int max_epoch, bool UI = 1);
        void SGD(double learning_rate, int max_epoch, bool UI = 1);
        void MBGD(double learning_rate, int max_epoch, int mini_batch_size, bool UI = 1);
        double score();
        void save(std::string fileName);
    private:

        void init();

        double Cost(std::vector<double> alpha, std::vector<std::vector<double>> X, std::vector<double> y);

        std::vector<double> Evaluate(std::vector<std::vector<double>> X);
        std::vector<double> propagate(std::vector<std::vector<double>> X);
        double Evaluate(std::vector<double> x);
        double propagate(std::vector<double> x);
        void forwardPass();

        void alphaProjection();

        double kernelFunction(std::vector<double> v, std::vector<double> u, std::string kernel);
        std::vector<std::vector<double>> kernelFunction(std::vector<std::vector<double>> U, std::vector<std::vector<double>> V, std::string kernel);

        std::vector<std::vector<double>> inputSet;
        std::vector<double> outputSet;
        std::vector<double> z;
        std::vector<double> y_hat;
        double bias;

        std::vector<double> alpha;
        std::vector<std::vector<double>> K;

        double C;
        int n;
        int k;

        std::string kernel;
        double p; // Poly
        double c; // Poly

        // UI Portion
        void UI(int epoch, double cost_prev);
    };
}
#endif //UNTITLED_DUALSVC_H
