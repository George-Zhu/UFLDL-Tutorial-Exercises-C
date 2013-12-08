#ifndef __SPARSEAECOST_H__
#define __SPARSEAECOST_H__

#include <iostream>
#include <Eigen/Dense>
#include "lbfgs.h"

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

struct instance
{
    int visibleSize;
    int hiddenSize;
    double lambda;
    double sparsityParam;
    double beta;
    MatrixXd data;
};

lbfgsfloatval_t sparseAECost(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    );

#endif