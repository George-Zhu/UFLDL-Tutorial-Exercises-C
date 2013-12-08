#ifndef __STACKEDAEPREDICT_H__
#define __STACKEDAEPREDICT_H__

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

MatrixXd stackedAEPredict(
	double* theta, 
	int numLayer, 
	int*layerSize, 
	int numClass, 
	MatrixXd& data);

#endif