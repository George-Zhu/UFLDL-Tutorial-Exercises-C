#include <iostream>
#include <Eigen/Dense>
#include <engine.h>
#include <mat.h>

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

int loadMat(MatrixXd& data, const char* nameFile) 
{
	MATFile *pFile;
	const char *nameArray;
	mxArray *pArray;

	pFile = matOpen(nameFile, "r");
	if (pFile == NULL)
	{
		cout << "Error opening file" << nameFile << endl;
		return 1;
	} 

	pArray = matGetNextVariable(pFile, &nameArray);
	cout << nameArray << endl; 

	int nrow = mxGetM(pArray);
	int ncol = mxGetN(pArray);
	data = MatrixXd::Zero(nrow, ncol);
	double *pData=(double*)mxGetPr(pArray); 
	for (int j=0; j<ncol; j++)
	{	
		for (int i=0; i<nrow; i++)
		{
			data(i, j) = (*pData);
			pData++;
		}
	}

	matClose(pFile);
	return 0;
}