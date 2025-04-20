#include <iostream>
#include "svd_main.h"
#include <Eigen/Dense>

int main()
{
    using namespace std;
    using namespace Eigen;
    Eigen::Matrix<float, Dynamic, Dynamic> A(16,15);
    
    A << float(1.2345), float(-98.3), float(0.0001), float(42.42), float(-7.77), float(3.1415), float(2.718), float(999.999), float(-1.111), float(0.005), float(123.456), float(-999.0), float(0.111), float(-77.1), float(6.66),
    float(0.00123), float(56789.1), float(-0.99), float(10.10), float(-42.0), float(5.4321), float(-0.618), float(8.8), float(-2.2), float(3.3), float(0.007), float(11.11), float(-5432.1), float(9.99), float(-7.77),
    float(0.314), float(15.92), float(-26.53), float(5.8), float(7.9), float(12.34), float(-55.5), float(1.1), float(-12.1), float(5.6), float(8.9), float(-1.2), float(3.4), float(7.7), float(6.5),
    float(-100.0), float(0.01), float(-0.1), float(0.001), float(9999.9), float(-42.42), float(0.789), float(-2.3456), float(8.123), float(-6.9), float(4.3), float(5.5), float(7.9), float(-33.1), float(2.4),
    float(88.88), float(-7.7), float(0.0067), float(4.2), float(-9.9), float(999.0), float(-0.101), float(44.4), float(-3.14), float(22.22), float(5.1), float(-3.2), float(14.15), float(4.44), float(1.23),
    float(-42.0), float(2.5), float(11.1), float(-0.4321), float(7.77), float(-5555.5), float(0.01), float(1.2), float(-8.8), float(3.9), float(-0.00456), float(99.99), float(-0.0123), float(6.2), float(4.6),
    float(-12.34), float(56.78), float(123.456), float(-999.0), float(0.98765), float(-0.1111), float(77.77), float(-88.88), float(99.99), float(-123.456), float(4321.0), float(-76.1), float(7.1), float(5.2), float(9.8),
    float(0.314), float(-27.1828), float(3.16), float(-5.1), float(1.414), float(2.236), float(-3.14), float(9.8), float(-7.1), float(0.987), float(3.2), float(-1.6), float(88.1), float(12.3), float(1.8),
    float(77.0), float(-7.89), float(55.5), float(6.9), float(-12.0), float(4.3), float(8.88), float(-1.12), float(3.9), float(-0.9), float(2.56), float(7.65), float(-9.4), float(8.4), float(6.12),
    float(0.123), float(-5.678), float(89.9), float(-4.3), float(1.02), float(-9.03), float(2.7), float(-3.5), float(8.14), float(4.5), float(9.87), float(-0.99), float(0.543), float(10.1), float(-1.9),
    float(123.1), float(-33.3), float(72.5), float(8.91), float(-5.2), float(6.4), float(-0.42), float(10.6), float(5.5), float(-7.3), float(0.345), float(-12.0), float(2.5), float(-14.2), float(4.7),
    float(9.1), float(8.5), float(-6.6), float(3.3), float(-2.2), float(4.4), float(-6.1), float(2.3), float(5.5), float(-6.3), float(8.9), float(3.2), float(-1.8), float(7.6), float(4.3),
    float(11.11), float(-0.123), float(2.718), float(3.14), float(1.01), float(7.77), float(-6.66), float(5.12), float(-3.45), float(2.78), float(-4.8), float(6.92), float(8.11), float(-2.11), float(9.32),
    float(14.14), float(-8.99), float(3.32), float(-7.99), float(4.56), float(3.33), float(-12.44), float(5.5), float(1.67), float(9.8), float(-11.1), float(7.8), float(5.6), float(-9.4), float(6.78),
    float(12.23), float(-4.45), float(2.13), float(7.43), float(3.21), float(8.21), float(-5.32), float(3.43), float(9.32), float(1.45), float(-3.92), float(11.2), float(7.7), float(-2.72), float(8.45),
    float(2.43), float(8.99), float(-3.11), float(4.56), float(-7.89), float(12.1), float(5.67), float(-6.78), float(4.32), float(-2.12), float(1.13), float(6.7), float(-9.22), float(8.33), float(3.31);

    Eigen::BDCSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    SVD<float,16,15> Ans(A);

    cout << Ans.matrixU() * Ans.singularValues() * Ans.matrixV().transpose() << "\n" << "\n";
    cout << Ans.matrixU() << "\n";
    cout << Ans.matrixV() << "\n";
    cout << Ans.singularValues() << "\n" << "\n";

   
    Array<float,1, Dynamic> sigm(15);
    sigm  = svd.singularValues();
    Eigen::Matrix<float, Dynamic, Dynamic > I(16, 15);
    I.setZero();
    I.block(0,0,15,15) = sigm.matrix().asDiagonal();
    Eigen::Matrix<float, Dynamic, Dynamic > U = svd.matrixU();
    cout << U*I* svd.matrixV().transpose() << "\n" << "\n";

    return 0;
}

// g++ -Wa,-mbig-obj -O2 -IC:/cpp/eigen-3.4.0 svd_test.cpp -o svd_test.exe
// svd_test.exe