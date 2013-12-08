#include <iostream>
#include <Eigen/Dense>
#include "lbfgs.h"
#include "stackClass.h"

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

void layerParam::init(int sizeLeft, int sizeRight, const double* theta)
{
    if (sizeLeft<0 || sizeRight<0)
    {
        cout << "Layer Parameters should be positive." << endl;
        return;
    }
    m_w = MatrixXd::Zero(sizeRight, sizeLeft);
    m_b = VectorXd::Zero(sizeRight);
    for (int i=0; i<sizeLeft*sizeRight; i++)
    {
        m_w.data()[i] = theta[i];
    }
    for (int j=0; j<sizeRight; j++)
    {
        m_b.data()[j] = theta[sizeLeft*sizeRight+j];
    }
}

layerParam::~layerParam()
{

}


stackClass::stackClass()
{
    m_numlayer = 0;
    m_hiddenLayerSize = NULL;
    m_theta = NULL;
    m_hiddenLayer = NULL;
}

stackClass::stackClass(int numlayer, const int* layerSize, const double* theta)
{
    m_numlayer = numlayer;
    m_hiddenLayerSize = layerSize;
    m_theta = theta;
    if (numlayer < 2)
    {
        m_hiddenLayer = NULL;
        cout << "Number of layers is not enough to make up a stack." << endl;
    }
    else
    {
        m_hiddenLayer = new layerParam[numlayer-1];
        int cnt = 0;
        for (int i=0; i<numlayer-1; i++)
        {
            m_hiddenLayer[i].init(layerSize[i], layerSize[i+1], theta+cnt);
            cnt += layerSize[i]*layerSize[i+1] + layerSize[i+1];
        }
    }

}

stackClass::~stackClass()
{
    if (m_numlayer < 2)
    {
        return;
    }
    delete [] m_hiddenLayer;
}