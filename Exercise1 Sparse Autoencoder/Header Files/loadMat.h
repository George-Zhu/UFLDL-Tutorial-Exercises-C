#ifndef __LOADMAT_H__
#define __LOADMAT_H__

#include <iostream>
#include <Eigen/Dense>
#include <engine.h>
#include <mat.h>

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

int loadMat(MatrixXd& data, const char* nameFile);

#endif