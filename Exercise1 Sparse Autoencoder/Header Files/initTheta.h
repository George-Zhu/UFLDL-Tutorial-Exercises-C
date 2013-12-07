#ifndef __INITPARAM_H__
#define __INITPARAM_H__

#include <iostream>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

void initTheta(int visibleSize, int hiddenSize, double* theta);

#endif