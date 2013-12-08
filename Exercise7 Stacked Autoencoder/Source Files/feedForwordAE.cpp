#include <iostream>
#include <Eigen/Dense>
#include "mathFunc.h"

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture;

MatrixXd feedForwardAE(
    double* theta, 
    int visibleSize, 
    int hiddenSize, 
    const MatrixXd& data)
{
    MatrixXd w1(hiddenSize, visibleSize);
    VectorXd b1(hiddenSize);

    for (int i=0; i<hiddenSize*visibleSize; i++)
    {
        w1.data()[i] = theta[i];
    }
    for (int i=0; i<hiddenSize; i++)
    {
        b1.data()[i] = theta[i+hiddenSize*visibleSize];
    }

    int ndata = data.cols();
    MatrixXd z2 = w1 * data + b1.replicate(1, ndata);
    MatrixXd a2 = sigmoid(z2);
    return a2;
}