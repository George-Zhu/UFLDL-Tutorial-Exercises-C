#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;  
using namespace Eigen::internal;  
using namespace Eigen::Architecture; 

MatrixXd sigmoid(const MatrixXd& input)
{
     int nrow = input.rows();
     int ncol = input.cols();
     MatrixXd output(nrow, ncol);
     MatrixXd temp = -input;
     output = 1 / (1 + temp.array().exp());
     return output;
}


MatrixXd sigmoidGrad(const MatrixXd& input)
{
    int nrow = input.rows();
    int ncol = input.cols();
    MatrixXd output(nrow, ncol);
    MatrixXd temp = (-input).array().exp();
    output = temp.array() / (1 + temp.array()).pow(2);
    return output;
}


