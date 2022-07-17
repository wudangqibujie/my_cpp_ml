//
// Created by Administrator on 2022/7/17 0017.
//

#ifndef AMG_LOGISTICREGRESSION_H
#define AMG_LOGISTICREGRESSION_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "assert.h"
//https://blog.csdn.net/qrqpjxq/article/details/77414898
struct DataSet{
    int numData;
    int numFeature;
    std::vector<std::vector<float>> features;
    std::vector<int> labels;
};

class LogisticRegression{
public:
    LogisticRegression(int max_iter, float learn_rate, float tol);
    ~LogisticRegression();
    DataSet loadData(std::string  filename);
    void initWeight(int length);
    std::vector<float> oneSampleGradient(std::vector<float> feature, int label);

};



#endif //AMG_LOGISTICREGRESSION_H
