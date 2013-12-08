#include <iostream>
#include <Eigen/Dense>
#include "mathFunc.h"
#include "sparseAECost.h"

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

lbfgsfloatval_t sparseAECost(
    void* netParam,
    const lbfgsfloatval_t *ptheta,
    lbfgsfloatval_t *grad,
    const int n,
    const lbfgsfloatval_t step
    )
{
    instance* pStruct = (instance*)(netParam);
    int hiddenSize = pStruct->hiddenSize;
    int visibleSize = pStruct->visibleSize;
    double lambda = pStruct->lambda;
    double beta = pStruct->beta;
    double sp = pStruct->sparsityParam;
    MatrixXd& data = pStruct->data;
    int ndim = data.rows();
    int ndata = data.cols();
    double cost = 0;

    MatrixXd w1(hiddenSize, visibleSize);
    MatrixXd w2(visibleSize, hiddenSize);
    VectorXd b1(hiddenSize);
    VectorXd b2(visibleSize);

    for (int i=0; i<hiddenSize*visibleSize; i++)
    {
        *(w1.data()+i) = *ptheta;
        ptheta++;
    }
    for (int i=0; i<visibleSize*hiddenSize; i++)
    {
        *(w2.data()+i) = *ptheta;
        ptheta++;
    }
    for (int i=0; i<hiddenSize; i++)
    {
        *(b1.data()+i) = *ptheta;
        ptheta++;
    }
    for (int i=0; i<visibleSize; i++)
    {
        *(b2.data()+i) = *ptheta;
        ptheta++;
    }

    MatrixXd z2 = w1 * data + b1.replicate(1, ndata);
    MatrixXd a2 = sigmoid(z2);
    MatrixXd z3 = w2 * a2 + b2.replicate(1, ndata);
    MatrixXd a3 = sigmoid(z3);

    VectorXd rho = a2.rowwise().sum() / ndata;
    VectorXd sparsityDelta = -sp / rho.array() + (1 - sp) / (1 - rho.array());

        MatrixXd delta3 = (a3 - data).array() * sigmoidGrad(z3).array();
        MatrixXd delta2 = (w2.transpose() * delta3 + beta * sparsityDelta.replicate(1, ndata)).array() 
                      * sigmoidGrad(z2).array();

    MatrixXd w1Grad = delta2 * data.transpose() / ndata + lambda * w1;
    VectorXd b1Grad = delta2.rowwise().sum() / ndata;
    MatrixXd w2Grad = delta3 * a2.transpose() / ndata + lambda * w2;
        VectorXd b2Grad = delta3.rowwise().sum() / ndata;

    cost = (0.5 * (a3 - data).array().pow(2)).matrix().sum() / ndata
            + 0.5 * lambda * ((w1.array().pow(2)).matrix().sum() 
            + (w2.array().pow(2)).matrix().sum())
            + beta * (sp * (sp / rho.array()).log() 
            + (1 - sp) * ((1 - sp) / (1 - rho.array())).log() ).matrix().sum();

    double* pgrad = grad;
    for (int i=0; i<hiddenSize*visibleSize; i++)
    {
        *pgrad = *(w1Grad.data()+i);
        pgrad++;
        
    }
    for (int i=0; i<visibleSize*hiddenSize; i++)
    {
        *pgrad = *(w2Grad.data()+i);
        pgrad++;
    }
    for (int i=0; i<hiddenSize; i++)
    {
        *pgrad = *(b1Grad.data()+i);
        pgrad++;
    }
    for (int i=0; i<visibleSize; i++)
    {
        *pgrad = *(b2Grad.data()+i);
        pgrad++;
    }

    return cost;
}
