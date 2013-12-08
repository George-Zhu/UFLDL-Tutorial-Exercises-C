#ifndef __STACKCLASS_H__
#define __STACKCLASS_H__

#include <iostream>
#include <Eigen/Dense>
#include "lbfgs.h"

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

class layerParam 
{
public:
	MatrixXd m_w;
	VectorXd m_b;
public:
	void init(int sizeLeft, int sizeRight, const double* theta);
	~layerParam();
};


class stackClass
{
public:
	int m_numlayer;
	const int* m_hiddenLayerSize;
	const double* m_theta;
	layerParam* m_hiddenLayer;
public:
	stackClass();
	stackClass(int numlayer, const int*layerSize, const double *theta);
	~stackClass();
};

#endif