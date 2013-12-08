#ifndef __SOFTMAXTRAIN_H__
#define __SOFTMAXTRAIN_H__

#include <iostream>
#include <Eigen/Dense>
#include "lbfgs.h"

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture;

struct instanceSF
{
    int inputSize;
    int numClass;
    double lambda;
    MatrixXd& label;
    MatrixXd& data;
};

lbfgsfloatval_t softmaxCost(
    void* netParam,
    const lbfgsfloatval_t *ptheta,
    lbfgsfloatval_t *grad,
    const int ntheta,
    const lbfgsfloatval_t step);

#endif