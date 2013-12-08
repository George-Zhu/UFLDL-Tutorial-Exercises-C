#include <iostream>
#include <Eigen/Dense>
#include <time.h>
#include "softmaxTrain.h"
#include "loadMat.h"
#include "initTheta.h"
#include "softmaxCost.h"
#include "sparseAECost.h"
#include "lbfgs.h"
#include "feedForwardAE.h"
#include "stackedAECost.h"
#include "stackedAEPredict.h"

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

static int debug();
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
    );


int main()
{                
    //Initialize parameters for training
    int numLayer = 3;
    int layerSize[3] = {28*28, 200, 200};
    int numClass = 10;
    double sparsityParam = 0.1;
    double lambda = 0.003;
    double beta = 3.0;
    int maxIter = 400;

    //Load data
    MatrixXd trainData;
    MatrixXd trainLabel;
    MatrixXd testData;
    MatrixXd testLabel;
    char* nameFile = "trainData.mat";
    if (loadMat(trainData, nameFile))
    {
        cout << "Fail to load file " << nameFile <<"." << endl;
        return 1;
    }
    nameFile = "trainLabel.mat";
    if (loadMat(trainLabel, nameFile))
    {
        cout << "Fail to load file " << nameFile <<"." << endl;
        return 1;
    }
    nameFile = "testData.mat";
    if (loadMat(testData, nameFile))
    {
        cout << "Fail to load file " << nameFile <<"." << endl;
        return 1;
    }
    nameFile = "testLabel.mat";
    if (loadMat(testLabel, nameFile))
    {
        cout << "Fail to load file " << nameFile <<"." << endl;
        return 1;
    }

    time_t trainTime = time(NULL);

    //Training for layer1
    #pragma region train_layer1
    cout << "******************Training for Layer1******************" << endl;
    instanceSP netParam1 = {layerSize[0], layerSize[1], lambda, sparsityParam, beta, trainData};
    int numParam = 2 * netParam1.hiddenSize * netParam1.visibleSize 
                     + netParam1.hiddenSize + netParam1.visibleSize;
    lbfgsfloatval_t cost;
    lbfgsfloatval_t* sae1Theta = lbfgs_malloc(numParam);
    lbfgs_parameter_t optParam;
    
    if (sae1Theta == NULL) 
    {
        cout << "ERROR: Failed to allocate a memory block for parameters." << endl;
        return 1;
    }
    initTheta(netParam1.hiddenSize, netParam1.visibleSize, sae1Theta);

    lbfgs_parameter_init(&optParam);
    optParam.max_iterations = maxIter;

    int ret = lbfgs(numParam, sae1Theta, &cost, sparseAECost, progress, (void*)(&netParam1), &optParam);
    cout << "L-BFGS optimization terminated with status code " << ret << endl << endl;

    MatrixXd sae1Feature = feedForwardAE(sae1Theta, netParam1.visibleSize, netParam1.hiddenSize, netParam1.data);
    #pragma endregion 
    
    //Training for layer2
    #pragma region train_layer2
    cout << "******************Training for Layer2******************" << endl;
    instanceSP netParam2 = {layerSize[1], layerSize[2], lambda, sparsityParam, beta, sae1Feature};
    
    numParam = 2 * netParam2.hiddenSize * netParam2.visibleSize 
                     + netParam2.hiddenSize + netParam2.visibleSize;

    lbfgsfloatval_t* sae2Theta = lbfgs_malloc(numParam);
    
    if (sae2Theta == NULL) 
    {
        cout << "ERROR: Failed to allocate a memory block for parameters." << endl;
        return 1;
    }
    initTheta(netParam2.hiddenSize, netParam2.visibleSize, sae2Theta);

    ret = lbfgs(numParam, sae2Theta, &cost, sparseAECost, progress, (void*)(&netParam2), &optParam);
    cout << "L-BFGS optimization terminated with status code " << ret << endl << endl;

    MatrixXd sae2Feature = feedForwardAE(sae2Theta, netParam2.visibleSize, netParam2.hiddenSize, netParam2.data);
    #pragma endregion 

    //Training for softmax classifier
    #pragma region train_softmax
    cout << "**************Training for Softmax Model***************" << endl;    
    double* saeSoftmaxTheta = softmaxTrain(layerSize[2], numClass, lambda, maxIter, sae2Feature, trainLabel);
    #pragma endregion

    //Get test accuracy before finetuning the whole network
    #pragma region acc_before_finetuning
    numParam = layerSize[2] * numClass 
               + layerSize[0] * layerSize[1] + layerSize[1] 
               + layerSize[1] * layerSize[2] + layerSize[2];

    lbfgsfloatval_t* stackedAETheta = lbfgs_malloc(numParam);    
    if (stackedAETheta == NULL) 
    {
        cout << "ERROR: Failed to allocate a memory block for parameters." << endl;
        return 1;
    }
    double* ptheta = stackedAETheta;
    for (int i=0; i<layerSize[2]*numClass; i++)
    {
        *ptheta = saeSoftmaxTheta[i];
        ptheta++;
    }
    for (int j=0; j<layerSize[0]*layerSize[1]; j++)
    {
        *ptheta = sae1Theta[j];
        ptheta++;
    }
    for (int k=0; k<layerSize[1]; k++)
    {
        *ptheta = sae1Theta[2*layerSize[0]*layerSize[1]+k];
        ptheta++;
    }
    for (int m=0; m<layerSize[1]*layerSize[2]; m++)
    {
        *ptheta = sae2Theta[m];
        ptheta++;
    }
    for (int n=0; n<layerSize[2]; n++)
    {
        *ptheta = sae2Theta[2*layerSize[1]*layerSize[2]+n];
        ptheta++;
    }

    MatrixXd predict = stackedAEPredict(stackedAETheta, numLayer, layerSize, numClass, testData);
    int cnt = 0;
    for (int i=0; i<testLabel.rows(); i++)
    {
        if (predict.data()[i] == testLabel.data()[i])
        {
            cnt++;
        }
    }
    double  accBefore = (double)(cnt) / testLabel.rows() * 100;
    #pragma endregion
    
    //Finetune softmax model
    #pragma region finetune
    cout << "****************Finetune Softmax Model*****************" << endl;
    lambda = 0.0001;
    instanceST softmaxParam = {numLayer, layerSize, numClass, lambda, trainLabel, trainData};
    ret = lbfgs(numParam, stackedAETheta, &cost, stackedAECost, progress, (void*)(&softmaxParam), &optParam);
    cout << "L-BFGS optimization terminated with status code " << ret << endl << endl;
    #pragma endregion

    trainTime = time(NULL) - trainTime;

    //Get test accuracy after finetuning the whole network
    #pragma region acc_after_finetuning
    predict = stackedAEPredict(stackedAETheta, numLayer, layerSize, numClass, testData);
    cnt = 0;
    for (int i=0; i<testLabel.rows(); i++)
    {
        if (predict.data()[i] == testLabel.data()[i])
        {
            cnt++;
        }
    }
    double accAfter = (double)(cnt) / testLabel.rows() * 100;
    #pragma endregion

    cout << "Before Finetuning Test Accuracy: " << accBefore << "%." << endl;
    cout << "After Finetuning Test Accuracy: " << accAfter << "%." << endl;
    cout << "Total time: " << trainTime << " seconds. " << endl;
    
    lbfgs_free(sae1Theta);
    lbfgs_free(sae2Theta);
    lbfgs_free(saeSoftmaxTheta);
    lbfgs_free(stackedAETheta);

    return 0;
}


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


static int debug()
{
    MatrixXd labelt(5,1);
    labelt<<0,1,2,3,4;
    cout << labelt << endl;
    MatrixXd datat = MatrixXd::Identity(5,5);
    cout << datat << endl;
    double* gradt = new double[220];
    int nthetat = 220;
    lbfgsfloatval_t* thetat = lbfgs_malloc(nthetat);
    int layerSizet[3] = {5,10,10};
    instanceST netParamt = {3,layerSizet,5,0.3,labelt,datat};

    for (int i=0; i<nthetat; i+=2)
    {
        thetat[i] = 0.1;
        thetat[i+1] = 0.2;
    }
    MatrixXd featuret;
    //    featuret = feedForwardAE(thetat, 5, 5, datat);
    //    softmaxCost((void*)&netParamt, thetat, gradt, nthetat, 0);
    //    stackedAECost((void*)&netParamt, thetat, gradt, nthetat, 0);
    //    stackedAEPredict(thetat, 3, layerSizet, 5, datat);
    delete [] gradt;
    lbfgs_free(thetat);
    return 0;
}