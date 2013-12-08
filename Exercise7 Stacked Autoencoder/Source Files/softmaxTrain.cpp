#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include "lbfgs.h"
#include "mathFunc.h"
#include "softmaxCost.h"

using namespace std;
using namespace cv;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

static int progress(
    void *instance,
    const lbfgsfloatval_t *theta,
    const lbfgsfloatval_t *grad,
    const lbfgsfloatval_t cost,
    const lbfgsfloatval_t normTheta,
    const lbfgsfloatval_t normGrad,
    const lbfgsfloatval_t step,
    int nparam,
    int niter,
    int ls
    )
{
    cout << "Iteration: " << niter <<"    cost: " << cost 
        << "	step: " << step << endl;
    return 0;
}


double* softmaxTrain(
    int inputSize, 
    int numClass, 
    double lambda, 
    int maxIter,
    MatrixXd& data, 
    MatrixXd& label)
{
    int numParam = numClass * inputSize;
    instanceSF netParam = {inputSize, numClass, lambda, label, data};
    lbfgsfloatval_t cost;
    lbfgsfloatval_t* theta = lbfgs_malloc(numParam);
    lbfgs_parameter_t optParam;
    lbfgs_parameter_init(&optParam);
    optParam.max_iterations = maxIter;

    if (theta == NULL) {
        cout << "ERROR: Failed to allocate a memory block for parameters." << endl;
        return NULL;
    }
    Mat randnArray(1, numClass*inputSize, CV_64FC1);
    randn(randnArray, 0, 1);
    double* prand = (double*)(randnArray.data);
    for (int i=0; i<numParam; i++)
    {
        theta[i] = 0.005 * prand[i];
    }

    int ret = lbfgs(numParam, theta, &cost, softmaxCost, progress, (void*)(&netParam), &optParam);
    cout << "L-BFGS optimization terminated with status code " << ret << endl << endl;

    return theta;
}