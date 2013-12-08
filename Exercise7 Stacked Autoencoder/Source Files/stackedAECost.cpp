#include <iostream>
#include <Eigen/Dense>
#include "lbfgs.h"
#include "mathFunc.h"
#include "stackedAECost.h"
#include "stackClass.h"

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

lbfgsfloatval_t stackedAECost(
    void* netParam,
    const lbfgsfloatval_t *ptheta,
    lbfgsfloatval_t *grad,
    const int ntheta,
    const lbfgsfloatval_t step)
{
    instanceST* pStruct = (instanceST*)(netParam);
    int numLayer = pStruct->numLayer;
    int* layerSize = pStruct->layerSize;
    int numClass = pStruct->numClass;
    double lambda = pStruct->lambda;
    MatrixXd& data = pStruct->data;
    MatrixXd& label = pStruct->label;
    double cost = 0;
    int ndata = data.cols();

    MatrixXd softmaxTheta(numClass, layerSize[numLayer-1]);
    for (int i=0; i<numClass*layerSize[numLayer-1]; i++)
    {
        softmaxTheta.data()[i] = ptheta[i];
    }

    MatrixXd groundTruth = MatrixXd::Zero(numClass, ndata);
    for (int i=0; i<ndata; i++)
    {
        groundTruth((int)(label(i)), i) = 1.0;
    }
    
    stackClass stack(numLayer, layerSize, ptheta+numClass*layerSize[numLayer-1]); 

    MatrixXd z2 = stack.m_hiddenLayer[0].m_w * data + stack.m_hiddenLayer[0].m_b.replicate(1, ndata);
    MatrixXd a2 = sigmoid(z2);
    MatrixXd z3 = stack.m_hiddenLayer[1].m_w * a2 + stack.m_hiddenLayer[1].m_b.replicate(1, ndata);
    MatrixXd a3 = sigmoid(z3);
    MatrixXd z4 = softmaxTheta * a3;
    MatrixXd a4 = z4.array().exp() / ((z4.array().exp().matrix().colwise().sum()).replicate(numClass, 1)).array();
    
    MatrixXd delta4 = a4 - groundTruth;
    MatrixXd delta3 = (softmaxTheta.transpose() * delta4).array() * sigmoidGrad(z3).array();
    MatrixXd delta2 = (stack.m_hiddenLayer[1].m_w.transpose() * delta3).array() * sigmoidGrad(z2).array();
    MatrixXd softmaxThetaGrad = -(1.0/ndata) * (groundTruth - a4) * a3.transpose() + lambda * softmaxTheta;
    
    MatrixXd w2Grad = (1.0/ndata) * delta3 * a2.transpose() + lambda * stack.m_hiddenLayer[1].m_w;
    MatrixXd w1Grad = (1.0/ndata) * delta2 * data.transpose() + lambda * stack.m_hiddenLayer[0].m_w;
    VectorXd b2Grad = (1.0/ndata) * delta3.rowwise().sum();
    VectorXd b1Grad = (1.0/ndata) * delta2.rowwise().sum();
    
    cost = -(1.0/ndata) * (groundTruth.array() * a4.array().log()).matrix().sum()
           + (lambda/2) * softmaxTheta.cwiseAbs2().sum()
           + (lambda/2) * (stack.m_hiddenLayer[0].m_w.cwiseAbs2().sum()
           + stack.m_hiddenLayer[1].m_w.cwiseAbs2().sum());
    
    double* pgrad = grad;
    for (int i=0; i<layerSize[2]*numClass; i++)
    {
        *pgrad = softmaxThetaGrad.data()[i];
        pgrad++;
    }
    for (int j=0; j<layerSize[0]*layerSize[1]; j++)
    {
        *pgrad = w1Grad.data()[j];
        pgrad++;
    }
    for (int k=0; k<layerSize[1]; k++)
    {
        *pgrad = b1Grad(k);
        pgrad++;
    }
    for (int m=0; m<layerSize[1]*layerSize[2]; m++)
    {
        *pgrad = w2Grad.data()[m];
        pgrad++;
    }
    for (int n=0; n<layerSize[2]; n++)
    {
        *pgrad = b2Grad(n);
        pgrad++;
    }

    return cost;
}