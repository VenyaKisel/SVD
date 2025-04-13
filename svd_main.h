#include <iostream>
#include <iomanip>
#include <cmath>
#include <tuple>
#include <vector>


#if __has_include(<Eigen/LU>)
#include <Eigen/LU> 
#elif __has_include(<eigen3/Eigen/LU>)
#include <eigen3/Eigen/LU>
#endif


#if __has_include(<Eigen/SVD>)
#include <Eigen/SVD> 
#elif __has_include(<eigen3/Eigen/SVD>)
#include <eigen3/Eigen/SVD>
#endif

using namespace std;

template<typename Scalar, int M, int N>
class SVD;

template<typename Scalar, int M, int N>
class SVD{
    private:
        Eigen::Matrix<Scalar, M, M> U;
        Eigen::Matrix<Scalar, M, N> S;
        Eigen::Matrix<Scalar, N, N> V;

        void Set_U(Eigen::Matrix<Scalar, M, M> a){U = a;};
        void Set_S(Eigen::Matrix<Scalar, M, N> a){S = a;};
        void Set_V(Eigen::Matrix<Scalar, N, N> a){V = a;};

    protected:
        SVD RefSVD(const Eigen :: Matrix<Scalar, M, N>& A, const Eigen::Matrix<Scalar, M, M>& Ui, const Eigen::Matrix<Scalar, N, N>& Vi){
            const int m = A.rows();
            const int n = A.cols();

            if (A.rows() < A.cols())
            {
                cout << "Attention! Number of the rows must be greater or equal  than  number of the columns";
                SVD ANS;
                return ANS;
            }

            using matrix_dd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
            using matrix_mm = Eigen::Matrix<double, M, M>; //U
            using matrix_mn = Eigen::Matrix<double, M, N>; //A
            using matrix_nn = Eigen::Matrix<double, N, N>; //V


            matrix_mn Ad = A.template cast<double>();
            matrix_mm Ud = Ui.template cast<double>();
            matrix_nn Vd = Vi.template cast<double>();

            // Step 1: Compute temp matrixes: R, S, T

            matrix_mm R = matrix_dd::Constant(m, m, 1.0) - Ud.transpose() * Ud;
            matrix_nn S = matrix_dd::Constant(n, n, 1.0) - Vd.transpose() * Vd;
            matrix_mn T = Ud.transpose() * Ad * Vd;
            matrix_nn F11(n, n);
            F11.setZero();
            matrix_nn G;
            G.setZero();

            // Step 2 and 3: compute approximate singular values
            // and compute diagonal parts of F11 and G

            matrix_nn Sigma_n(n, n);
            Sigma_n.setZero();

            for(int i = 0; i < n; i++){
                Sigma_n(i, i) = T(i, i) / (1 - (R(i, i) + S(i, i)) * 0.5);
                F11(i, i) = R(i, i) * 0.5;
                G(i, i) = S(i, i) * 0.5; 
            }
            // Step 4: Huge step : compute off-diagonal parts of F11 and G

            double alpha;
            double betta;
            double sigma_i_sqr;
            double sigma_j_sqr;

            for(int i = 0; i < n; i++){
                sigma_i_sqr = Sigma_n(i, i) * Sigma_n(i, i);
                for(int j = 0; j < n; j++){
                    if(i != j){
                        sigma_j_sqr = Sigma_n(j, j) * Sigma_n(j, j);
                        alpha = T(i, j) + Sigma_n(j, j) * R(i, j);
                        betta = T(j, i) + Sigma_n(j, j) * S(i, j);

                        F11(i, j) = ((alpha * Sigma_n(j, j) + betta * Sigma_n(i, i)) / (sigma_j_sqr - sigma_i_sqr));
                        G(i, j) = ((alpha * Sigma_n(i, i) + betta * Sigma_n(j, j)) / (sigma_j_sqr - sigma_i_sqr));
                    }
                }
            }

            matrix_mn Sigma(m, n);
            Sigma.setZero();
            Sigma.block(0, 0, n, n) = Sigma_n.transpose();

            // Step 5: compute F12;

            matrix_dd F12(n, m - n);

            for (int i = 0; i < n; ++i) {
                for (int j = n; j < m; ++j) {
                    F12(i, j - n) = -T(j, i) / Sigma_n(i, i);
                }
            }
            

            // Step 6: compute F21

            matrix_dd F21(m - n, n);

            for (int i = n; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    int row = i - n;
                    int col = j;
                    F21(i - n, j) = R(i,  j) - F12(j, i - n);
                }
            }

            // Step 7: compute F22;
            matrix_dd F22(m - n, m - n);
            for(int i = n; i < m; ++i){
                for(int j = n; j < m; ++j){
                    F22(i - n, j - n) = R(i, j) * 0.5;
                }
            }
        
            // Step 8: convert matrixes and compute final answer

            matrix_mm F(m, m);
            F.block(0, 0, n, n) = F11;
            F.block(0, n, n, m - n) = F12;
            F.block(n, 0, m - n, n) = F21;
            F.block(n, n, m - n, m - n) = F22;

            matrix_mm U = Ud + Ud * F;
            matrix_nn V = Vd + Vd * G;

            SVD<Scalar, M, N> ans;
            ans.Set_U(U.template cast<Scalar>());
            ans.Set_V(V.template cast<Scalar>());
            ans.Set_S(Sigma.template cast<Scalar>());
            return ans;
        };

    public:
        SVD() {};

        SVD(Eigen::Matrix<Scalar, M, N> A)
        {
            Eigen::BDCSVD< Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
            SVD<Scalar, M, N> temp;
            temp = RefSVD(A, svd.matrixU(), svd.matrixV());
            this->U = temp.matrixU();
            this->V = temp.matrixV();
            this->S = temp.singularValues();
        }

        Eigen::Matrix<Scalar, N, N> matrixV()
        {
            return V;
        }

        Eigen::Matrix<Scalar, M, M> matrixU()
        {
            return U;
        }

        Eigen::Matrix<Scalar, M, N> singularValues()
        {
            return S;
        }
};
