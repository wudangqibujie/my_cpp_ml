#include <iostream>
#include "SVC/SVC.h"
#include "PCA/PCA.h"
#include "MLP/MLP.h"
#include "kNN/kNN.h"
#include "KMeans/KMeans.h"
#include <vector>
#include "ANN//ANN.h"
#include "Data/Data.h"
#include "LinAlg//LinAlg.h"
#include "AutoEncoder/AutoEncoder.h"
#include "Convolutions/Convolutions.h"
#include "Transforms/Transforms.h"
#include "LinReg/LinReg.h"

using namespace MLPP;

int main() {
    LinAlg alg;
    Data data;


//    std::cout << "Hello, ANN" << std::endl;
//    std::vector<std::vector<double>> inputSet = {{0,0,1,1}, {0,1,0,1}};
//    std::vector<double> outputSet = {0,1,1,0};
//    ANN ann(alg.transpose(inputSet), outputSet);
//    ann.addLayer(2, "Cosh");
//    ann.addOutputLayer("Sigmoid", "LogLoss");
//
////    ann.AMSGrad(0.1, 10000, 1, 0.9, 0.999, 0.000001, 1);
////    ann.Adadelta(1, 1000, 2, 0.9, 0.000001, 1);
////    ann.Momentum(0.1, 8000, 2, 0.9, true, 1);
//
//    ann.setLearningRateScheduler("Step", 0.5, 1000);
//    ann.gradientDescent(0.1, 30000);
//    alg.printVector(ann.modelSetTest(alg.transpose(inputSet)));
//    std::cout << "ACC:" << 100 * ann.score() << "%" << std::endl;

//    std::cout << "Hello, AutoEncoder" << std::endl;
//    std::vector<std::vector<double>> inputSet = {{1,2,3,4,5,6,7,8,9,10}, {3,5,9,12,15,18,21,24,27,30}};
//    AutoEncoder model(alg.transpose(inputSet), 5);
//    model.SGD(0.001, 300000, 0);
//    alg.printMatrix(model.modelSetTest(alg.transpose(inputSet)));
//    std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;

//    std::cout << "Hello, AutoEncoder" << std::endl;
//    Convolutions conv;
//    std::vector<std::vector<double>> input = {
//            {62,55,55,54,49,48,47,55},
//            {62,57,54,52,48,47,48,53},
//            {61,60,52,49,48,47,49,54},
//            {63,61,60,60,63,65,68,65},
//            {67,67,70,74,79,85,91,92},
//            {82,95,101,106,114,115,112,117},
//            {96,111,115,119,128,128,130,127},
//            {109,121,127,133,139,141,140,133},
//    };
//    Transforms trans;
//    alg.printMatrix(trans.discreteCosineTransform(input));
//    alg.printMatrix(conv.convolve(input, conv.getPrewittVertical(), 1)); // Can use padding
//    alg.printMatrix(conv.pool(input, 4, 4, "Max")); // Can use Max, Min, or Average pooling.
//    std::vector<std::vector<std::vector<double>>> tensorSet;
//    tensorSet.push_back(input);
//    tensorSet.push_back(input);
//    alg.printVector(conv.globalPool(tensorSet, "Average")); // Can use Max, Min, or Average global pooling.
//    std::vector<std::vector<double>> laplacian = {{1, 1, 1}, {1, -4, 1}, {1, 1, 1}};
//    alg.printMatrix(conv.convolve(conv.gaussianFilter2D(5, 1), laplacian, 1));

//    std::cout << "Hello, RegLin" << std::endl;
//    auto [inputSet, outputSet] = data.loadCaliforniaHousing();
//    LinReg model(inputSet, outputSet); // Can use Lasso, Ridge, ElasticNet Reg
//    model.gradientDescent(0.001, 30, 0);
//    model.SGD(0.00000001, 300000, 1);
//    model.MBGD(0.001, 10000, 2, 1);
//    model.normalEquation();
//    LinReg adamModel(alg.transpose(inputSet), outputSet);
//    alg.printVector(model.modelSetTest(inputSet));
//    std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;

//    std::cout << "Hello, Kmeans" << std::endl;
//    std::vector<std::vector<double>> inputSet = {{32, 0, 7}, {2, 28, 17}, {0, 9, 23}};
//    KMeans kmeans(inputSet, 3, "KMeans++");
//    kmeans.train(3, 1);
//    std::cout << std::endl;
//    alg.printMatrix(kmeans.modelSetTest(inputSet)); // Returns the assigned centroids to each of the respective training examples
//    std::cout << std::endl;
//    alg.printVector(kmeans.silhouette_scores());

//    std::cout << "Hello, KNN" << std::endl;
//    std::vector<std::vector<double>> inputSet = {{1,2,3,4,5,6,7,8}, {0,0,0,0,1,1,1,1}};
//    std::vector<double> outputSet = {0,0,0,0,1,1,1,1};
//    kNN knn(alg.transpose(inputSet), outputSet, 8);
//    alg.printVector(knn.modelSetTest(alg.transpose(inputSet)));
//    std::cout << "ACCURACY: " << 100 * knn.score() << "%" << std::endl;

//    std::cout << "Hello, MLP" << std::endl;
//    std::vector<std::vector<double>> inputSet = {{0,0,1,1}, {0,1,0,1}};
//    inputSet = alg.transpose(inputSet);
//    std::vector<double> outputSet = {0,1,1,0};
//    MLP model(inputSet, outputSet, 2);
//    model.gradientDescent(0.1, 10000, 0);
//    alg.printVector(model.modelSetTest(inputSet));
//    std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;


//    std::cout << "Hello, PCA" << std::endl;
//    std::vector<std::vector<double>> inputSet = {{1,1}, {1,1}};
//    auto [Eigenvectors, Eigenvalues] = alg.eig(inputSet);
//    std::cout << "Eigenvectors:" << std::endl;
//    alg.printMatrix(Eigenvectors);
//    std::cout << std::endl;
//    std::cout << "Eigenvalues:" << std::endl;
//    alg.printMatrix(Eigenvalues);

    std::cout << "Hello, SVC" << std::endl;
    auto [inputSet, outputSet] = data.loadBreastCancerSVC();
    SVC model(inputSet, outputSet, 1);
    model.SGD(0.00001, 100000, 1);
    alg.printVector(model.modelSetTest(inputSet));
    std::cout << "ACCURACY: " << 100 * model.score() << "%" << std::endl;


    return 0;
}
