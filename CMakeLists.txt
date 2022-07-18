cmake_minimum_required(VERSION 3.21)
project(AMG)

set(CMAKE_CXX_STANDARD 17)
include_directories(D:\\eigen-3.3.8\\Eigen)
#add_executable(untitled main.cpp ANN/ANN.cpp Cost/Cost.cpp Data/Data.cpp HiddenLayer/HiddenLayer.cpp LinAlg/LinAlg.cpp OutputLayer/OutputLayer.cpp Regularization/Reg.cpp SoftmaxNet/SoftmaxNet.cpp Stat/Stat.cpp Utilities/Utilities.cpp HiddenLayer/HiddenLayer.cpp LinAlg/LinAlg.cpp OutputLayer/OutputLayer.cpp Regularization/Reg.cpp SoftmaxNet/SoftmaxNet.cpp Stat/Stat.cpp Utilities/Utilities.cpp Activation/Activation.cpp AutoEncoder/AutoEncoder.h AutoEncoder/AutoEncoder.cpp Convolutions/Convolutions.h Convolutions/Convolutions.cpp Transforms/Transforms.h Transforms/Transforms.cpp DualSVC/DualSVC.h DualSVC/DualSVC.cpp LinReg/LinReg.h LinReg/LinReg.cpp KMeans/KMeans.h KMeans/KMeans.cpp kNN/kNN.h kNN/kNN.cpp MLP/MLP.h MLP/MLP.cpp PCA/PCA.h PCA/PCA.cpp SVC/SVC.h SVC/SVC.cpp mit.cpp)
add_executable(bmw mit.cpp Easy_LR/logisticregression.h Easy_LR/common.h Easy_LR/logisticregression.cpp)