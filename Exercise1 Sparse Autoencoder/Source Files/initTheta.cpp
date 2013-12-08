#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

void initTheta(int visibleSize, int hiddenSize, double* theta)
{
    double range = std::sqrt(6.0) / std::sqrt(hiddenSize+visibleSize+1.0);
    VectorXd w1(hiddenSize * visibleSize);
    VectorXd w2(visibleSize * hiddenSize);
    VectorXd b1(hiddenSize);
    VectorXd b2(visibleSize);

    w1 = VectorXd::Random(hiddenSize * visibleSize) * range; 
    w2 = VectorXd::Random(visibleSize * hiddenSize) * range; 
    b1 = VectorXd::Zero(hiddenSize);
    b2 = VectorXd::Zero(visibleSize);

    double* ptheta = theta;
    for (int i=0; i<hiddenSize*visibleSize; i++)
    {
        *ptheta = w1.data()[i];
        ptheta++;
    }
    for (int i=0; i<visibleSize*hiddenSize; i++)
    {
        *ptheta = w2.data()[i];
        ptheta++;
    }
    for (int i=0; i<hiddenSize; i++)
    {
        *ptheta = b1.data()[i];
        ptheta++;
    }
    for (int i=0; i<visibleSize; i++)
    {
        *ptheta = b2.data()[i];
        ptheta++;
    }
}

