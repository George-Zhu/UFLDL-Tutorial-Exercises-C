#ifndef __DISPLAYNETWORK_H__
#define __DISPLAYNETWORK_H__

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

void displayNetwork(MatrixXd feature);

#endif