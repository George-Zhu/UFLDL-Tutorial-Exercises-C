#ifndef __FEEDWARDAE_H__
#define __FEEDWARDAE_H__

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture;

MatrixXd feedForwardAE(
    double* theta, 
    int visibleSize, 
    int hiddenSize, 
    const MatrixXd& data);


#endif

