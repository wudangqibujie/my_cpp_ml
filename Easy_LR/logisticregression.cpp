//
// Created by jay on 2022/7/18.
//
#include "logisticregression.h"
#include "common.h"
#include <random>
#include <chrono>

LogisticRegression::LogisticRegression(int max_iter, float learn_rate, float tol = 0.0001) {
    this -> maxIter_ = max_iter;
    this -> learnRate = learn_rate;
    this -> tol_ = tol;
}

LogisticRegression::~LogisticRegression() {

}

DataSet LogisticRegression::loadData(std::string filename) {
    std::ifstream ifile(filename);
    DataSet dataset;
    if (!ifile){
        std::cout << "can not open file" << filename << std::endl;
        return dataset;
    }
    std::string line;
    while (std::getline(ifile, line)){
        std::vector<std::string> tokens = common::Split(line, ',');
        std::vector<float> feature;
        for (int i = 0; i < tokens.size(); ++i) {
            if (i == tokens.size() - 1){
                dataset.labels.push_back(atoi(tokens[i].c_str()));
            }else{
                feature.push_back(atof(tokens[i].c_str()));
            }
        }
        dataset.features.push_back(feature);
        dataset.numData += 1;
    }
    dataset.numFeature = dataset.features[0].size();
    return dataset;
}


void LogisticRegression::initWeight(int length) {
    this -> weight.push_back(1.0);
    for (int i = 0; i < length; ++i) {
        this -> weight.push_back(1.0);
    }
}

std::vector<float> LogisticRegression::oneSampleGradient(std::vector<float> feature, int label) {
    std::vector<float> grident(this -> weight.size(), 0.0);
    float predY = predict_proba(feature);
    grident[0] = predY - label;
    for (int i = 0; i < feature.size(); ++i) {
        grident[i + 1] = (predY - label) * feature[i];
    }
    return grident;
}


void LogisticRegression::train(DataSet *dataset, std::string gdType) {
    DataSet* traindata = dataset;
    int dataNum = traindata -> numData;
    int featureNum = traindata -> numFeature;
    initWeight(featureNum);
    for (int iter = 0; iter < this -> maxIter_; ++iter) {
        if (gdType == "gd"){

        }
    }
}