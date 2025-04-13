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

template<typename T, int M, int N>
class SVD;

template<typename T, int M, int N>
class SVD{
    private:
        Eigen::Matrix<T, M, M> U;
        Eigen::Matrix<T, M, N> S;
        Eigen::Matrix<T, N, N> V;

        void Set_U(Eigen::Matrix<T, M, M> a){U = a;};
        void Set_S(Eigen::Matrix<T, M, N> a){S = a;};
        void Set_V(Eigen::Matrix<T, N, N> a){V = a;};

    protected:
        SVD RefSVD(const Eigen :: Matrix<T, M, N>& A, const Eigen::Matrix<T, M, M>& Ui, const Eigen::Matrix<T, N, N>& Vi){
            const int m = A.rows();
            const int n = A.cols();

            if (A.rows() < A.cols())
            {
                cout << "Attention! Number of the rows must be greater or equal  than  number of the columns";
                SVD_alg_1 ANS;
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
            matrix_mm F11(n, n);
            matrix_nn G;

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

            matrix_dd F12(m - n, n);

            for(int i = 0; i < n; i++){
                for(int j = n + 1; j < m; j++){
                    F12(i, j) = -T(j, i) / Sigma_n(j, j);
                }
            }

            // Step 6: compute F21

            matrix_dd F21(n, m - n);

            for(int i = n + 1; i <= m; i++){
                for(int j = 0; j < n; j++){
                    F21(i, j) = R(i, j) - F21(j, i);
                }
            }

            // Step 7: compute F22;
            matrix_dd F22(m - n, m - n);
            for(int i = n + 1; i < m; i++){
                for(int j = 0; j < m; j++){
                    F22(i, j) = R(i, j) * 0.5;
                }
            }
        
            // Step 8: convert matrixes and compute final answer

            matrix_mm F(m, m);
            F.block(0, 0, n, n) = F11;
            F.block(0, n, n, m - n) = F12;
            F.block(n, 0, m - n, n) = F21;
            F.block(n, n, m - n, m - n) = F22;

            matrix_mm U = Ud + Ud * F;
            matrix_nn V = Vd + Vd * F;

            SVD ans;
            ans.Set_U(U_new.template cast<T>());
            ans.Set_V(V_new.template cast<T>());
            ans.Set_S(Sigma_full.template cast<T>());
            return ans;
        };

    public:
        SVD() {};

        SVD_alg_1(Eigen::Matrix<T, M, N> A)
        {
            Eigen::BDCSVD< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
            SVD<T, M, N> temp;
            temp = RefSVD(A, svd.matrixU(), svd.matrixV());
            this->U = temp.matrixU();
            this->V = temp.matrixV();
            this->S = temp.singularValues();
        }

        Eigen::Matrix<T, N, N> matrixV()
        {
            return V;
        }

        Eigen::Matrix<T, M, M> matrixU()
        {
            return U;
        }

        Eigen::Matrix<T, M, N> singularValues()
        {
            return S;
        }
};
