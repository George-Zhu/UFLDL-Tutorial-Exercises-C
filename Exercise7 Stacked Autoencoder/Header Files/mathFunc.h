#ifndef __MATHFUNC_H__
#define __MATHFUNC_H__

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture;

MatrixXd sigmoid(const MatrixXd& input);
MatrixXd sigmoidGrad(const MatrixXd& input);

#endif