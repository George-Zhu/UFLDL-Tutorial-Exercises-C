#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

void displayNetwork(MatrixXd feature)
{
    int nrow = feature.rows();
    int ncol = feature.cols();
    int sz = (int)( std::sqrt((double)(nrow)) );
    int n = (int)( std::sqrt((double)(ncol)) );
    MatrixXd network(1+(sz+1)*n, 1+(sz+1)*n);
    network = MatrixXd::Zero(1+(sz+1)*n, 1+(sz+1)*n);
    MatrixXd temp(sz, sz);

	feature = feature.array() - feature.mean();

	for (int i=0; i<ncol; i++)
	{
		for (int j=0; j<nrow; j++)
		{
			temp.data()[j] = feature(j, i);
		}

		network.block(1+(sz+1)*(i/n), 1+(sz+1)*(i%n), sz, sz) 
					  = (temp.array() - feature.col(i).minCoeff())
					    / (feature.col(i).maxCoeff()-feature.col(i).minCoeff());
	}

	Mat image;
	image = Mat::zeros(network.rows(), network.cols(), CV_64FC1);
	double* pimage = (double*)(image.data);
	for (int i=0; i<network.rows(); i++)
	{
		for (int j=0; j<network.cols(); j++)
		{
			*pimage = network(i,j);
			pimage++;
		}
	}

 	namedWindow("Weight Map", 0);
 	imshow("Weight Map", image);
	image.convertTo(image, CV_8UC1, 255);
	imwrite("../Weight Map.jpg", image);
}
