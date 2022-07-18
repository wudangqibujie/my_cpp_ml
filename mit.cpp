//
// Created by Administrator on 2022/7/17 0017.
//

#include <ostream>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>

int main(){
    std::cout << "asdqw" << std::endl;
    Eigen::MatrixXd h(4, 4);
    h.setIdentity();
    std::cout << h << std::endl;
    return 0;
}