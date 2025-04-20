#include <iostream>
#include <Eigen/Dense>
#include <complex>

using namespace std;
using namespace Eigen;

int main()
{
    // Объявляем матрицу A размера 5x4 с комплексными значениями (std::complex<float>)
    Matrix<complex<float>, Dynamic, Dynamic> A(5,4);
    
    // Инициализируем матрицу A; все числа оборачиваем в float()
    A << complex<float>(float(1.0),  float(2.0)),   complex<float>(float(3.0),   float(-1.0)),  complex<float>(float(-4.5),  float(0.5)),   complex<float>(float(2.2),  float(-3.3)),
         complex<float>(float(0.0),  float(-1.0)),  complex<float>(float(5.5),   float(2.2)),   complex<float>(float(3.3),   float(3.3)),   complex<float>(float(-6.7), float(4.4)),
         complex<float>(float(2.1),  float(1.1)),   complex<float>(float(-3.4),  float(-2.5)),  complex<float>(float(4.2),   float(0.0)),   complex<float>(float(1.1),  float(-1.1)),
         complex<float>(float(7.8),  float(-3.3)),  complex<float>(float(-4.2),  float(4.1)),   complex<float>(float(0.0),   float(0.0)),   complex<float>(float(5.3),  float(-2.2)),
         complex<float>(float(-6.5), float(1.3)),   complex<float>(float(8.1),   float(-6.7)),  complex<float>(float(2.3),   float(-3.4)),  complex<float>(float(-4.6), float(0.7));

    // Вычисляем комплексное SVD матрицы A, используя JacobiSVD
    JacobiSVD< Matrix<complex<float>, Dynamic, Dynamic> > svd(A, ComputeFullU | ComputeFullV);

    // Определяем количество сингулярных значений (минимальная размерность матрицы A)
    int minDim = svd.singularValues().size();  // для A(5x4) minDim == 4

    // Формируем матрицу S размера 5x4: заполняем нулями и диагоаналём сингулярных значений
    Matrix<complex<float>, Dynamic, Dynamic> S = Matrix<complex<float>, Dynamic, Dynamic>::Zero(5,4);
    S.block(0, 0, minDim, minDim) = svd.singularValues().cast<complex<float>>().asDiagonal();

    // Восстанавливаем матрицу A по формуле: A = U * S * V^*
    Matrix<complex<float>, Dynamic, Dynamic> A_reconstructed = svd.matrixU() * S * svd.matrixV().adjoint();

    // Выводим исходную матрицу и восстановленную
    cout << "Исходная матрица A:\n" << A << "\n\n";
    cout << "Восстановленная матрица A (U * S * V^*):\n" << A_reconstructed << "\n\n";

    return 0;
}