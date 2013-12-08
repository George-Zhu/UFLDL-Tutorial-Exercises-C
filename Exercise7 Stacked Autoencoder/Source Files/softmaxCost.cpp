#include <iostream>
#include <Eigen/Dense>
#include "lbfgs.h"
#include "mathFunc.h"
#include "softmaxCost.h"

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture;

lbfgsfloatval_t softmaxCost(
    void* netParam,
    const lbfgsfloatval_t *ptheta,
    lbfgsfloatval_t *grad,
    const int ntheta,
    const lbfgsfloatval_t step)
{
    instanceSF* pStruct = (instanceSF*)(netParam);
    int inputSize = pStruct->inputSize;
    int numClass = pStruct->numClass;
    double lambda = pStruct->lambda;
    MatrixXd& data = pStruct->data;
    MatrixXd& label = pStruct->label;
    double cost = 0;
    int ndata = data.cols();

    MatrixXd theta(numClass, inputSize);
    MatrixXd thetaGrad(numClass,inputSize);
    for (int i=0; i<ntheta; i++)
    {
        theta.data()[i] = ptheta[i];
    }
    
    MatrixXd groundTruth = MatrixXd::Zero(numClass, ndata);
    for (int i=0; i<ndata; i++)
    {
        groundTruth((int)(label(i)), i) = 1;
    }

    MatrixXd z = theta * data;
    MatrixXd h = z.array().exp() / ((z.array().exp().matrix().colwise().sum()).replicate(numClass, 1)).array();
    
    cost = -(1.0/ndata) * (h.array().log() * groundTruth.array()).matrix().sum()
           + lambda / 2 * theta.cwiseAbs2().sum();
    thetaGrad =  -(1.0/ndata) * (groundTruth - h) * data.transpose() + lambda * theta;

    for (int i=0; i<ntheta; i++)
    {
        grad[i] = thetaGrad.data()[i];
    }

    return cost;
}