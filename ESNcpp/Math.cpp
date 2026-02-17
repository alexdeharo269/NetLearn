#include <Eigen/Dense>
#include <vector>    


//Stable math functios using Eigen. 
double computeSpectralRadius_Internal(const std::vector<std::vector<double>> &matrix)
{
    int n = matrix.size();
    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A(i, j) = matrix[i][j];
        }
    }
    Eigen::EigenSolver<Eigen::MatrixXd> es(A, false);
    auto evals = es.eigenvalues();
    double max_norm = 0.0;
    for (int i = 0; i < evals.size(); ++i)
    {
        max_norm = std::max(max_norm, std::abs(evals[i]));
    }
    return max_norm;
}