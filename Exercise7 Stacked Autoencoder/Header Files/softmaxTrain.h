#ifndef __SOFTMAXTRAIN_H__
#define __SOFTMAXTRAIN_H__

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 



double* softmaxTrain(
	int inputSize, 
	int numClass, 
	double lambda, 
	int maxIter,
	MatrixXd& data, 
    MatrixXd& label);

#endif