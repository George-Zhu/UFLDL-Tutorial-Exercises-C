#include <time.h>
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "loadMat.h"
#include "initTheta.h"
#include "sparseAECost.h"
#include "lbfgs.h"
#include "displayNetwork.h"

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
	cout << "Iteration: " << niter <<"	cost: " << cost 
		<< "	step: " << step << endl;
	return 0;
}

int main()
{
	//Initialize the parameters for network and cost function
	instance netParam;
	netParam.visibleSize = 8*8;   
	netParam.hiddenSize = 25;     
	netParam.sparsityParam = 0.01;  
	netParam.lambda = 0.0001;          
	netParam.beta = 3.0;            

	char* nameFile = "patches.mat";
	if (loadMat(netParam.data, nameFile))
	{
		cout << "Fail to load file " << nameFile <<"." << endl;
		return 1;
	}
	int maxIter = 400;
	int numParam = 2 * netParam.hiddenSize * netParam.visibleSize 
		             + netParam.hiddenSize + netParam.visibleSize;
	
	//Initialize the parameters for lbfgs
	lbfgsfloatval_t cost;
	lbfgsfloatval_t* theta = lbfgs_malloc(numParam);
	lbfgs_parameter_t optParam;
	if (theta == NULL) {
		cout << "ERROR: Failed to allocate a memory block for parameters." << endl;
		return 1;
	}
	initTheta(netParam.hiddenSize, netParam.visibleSize, theta);
	lbfgs_parameter_init(&optParam);
	optParam.max_iterations = maxIter;

	//Training
	time_t runTime = time(NULL);
	int ret = lbfgs(numParam, theta, &cost, sparseAECost, progress, (void*)(&netParam), &optParam);
	cout << "L-BFGS optimization terminated with status code " << ret << endl;
	runTime = time(NULL) - runTime;
	cout << "Total run time: " << runTime << "seconds." << endl; 

	//Generate the feature map and display 
	MatrixXd w1(netParam.hiddenSize, netParam.visibleSize);
	for (int i=0; i<netParam.hiddenSize*netParam.visibleSize; i++)
	{
		w1.data()[i] = theta[i];
	}
	MatrixXd feature(netParam.visibleSize, netParam.hiddenSize);
	feature = w1.transpose();
        displayNetwork(feature);
	waitKey(0);

	//Free Memory
	lbfgs_free(theta);	

	return 0;
}



