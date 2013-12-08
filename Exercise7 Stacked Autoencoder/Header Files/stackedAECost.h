#ifndef __STACKEDAECOST_H__ 
#define __STACKEDAECOST_H__

#include <iostream>
#include <Eigen/Dense>
#include "lbfgs.h"
#include "stackClass.h"

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

struct instanceST
{
	int numLayer;
	int* layerSize;
	int numClass;
	double lambda;
	MatrixXd& label;
	MatrixXd& data;
};

lbfgsfloatval_t stackedAECost(
	void* netParam,
	const lbfgsfloatval_t *ptheta,
	lbfgsfloatval_t *grad,
	const int ntheta,
	const lbfgsfloatval_t step);

#endif