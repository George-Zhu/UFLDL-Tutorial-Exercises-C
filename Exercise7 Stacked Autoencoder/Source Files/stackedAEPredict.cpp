#include <iostream>
#include <Eigen/Dense>
#include "mathFunc.h"
#include "stackClass.h"

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

MatrixXd stackedAEPredict(
    double* theta, 
    int numLayer, 
    int*layerSize, 
    int numClass, 
    MatrixXd& data)
{
    int ndata = data.cols();
    stackClass stack(numLayer, layerSize, theta+numClass*layerSize[numLayer-1]);

    MatrixXd softmaxTheta(numClass, layerSize[numLayer-1]);    
    for (int i=0; i<numClass*layerSize[numLayer-1]; i++)
    {
        softmaxTheta.data()[i] = theta[i];
    }

    MatrixXd z2 = stack.m_hiddenLayer[0].m_w * data + stack.m_hiddenLayer[0].m_b.replicate(1, ndata);
    MatrixXd a2 = sigmoid(z2);
    MatrixXd z3 = stack.m_hiddenLayer[1].m_w * a2 + stack.m_hiddenLayer[1].m_b.replicate(1, ndata);
    MatrixXd a3 = sigmoid(z3);
    MatrixXd z4 = softmaxTheta * a3;
    MatrixXd a4 = z4.array().exp() / ((z4.array().exp().matrix().colwise().sum()).replicate(numClass, 1)).array();
    
    MatrixXd predict(ndata, 1);
    for (int i=0; i<ndata; i++)
    {
        double tempValue = 0.0;
        double tempIndex = 0.0;
        for (int j=0; j<numClass; j++)
        {
            if (tempValue < a4(j,i))
            {
                tempValue = a4(j,i);
                tempIndex = j;
            }
        }
        predict.data()[i] = tempIndex;
    }

    return predict;
}