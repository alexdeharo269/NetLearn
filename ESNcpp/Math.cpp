#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <format>

using namespace std;

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

    // BLINDAJE: Comprobar si Eigen convergió
    if (es.info() != Eigen::Success)
    {
        std::cout << "ALERTA: EigenSolver no convergió al calcular autovalores." << std::endl;
        return 1.0; // Evita el crash
    }

    auto evals = es.eigenvalues();
    double max_norm = 0.0;
    for (int i = 0; i < evals.size(); ++i)
    {
        max_norm = std::max(max_norm, std::abs(evals[i]));
    }
    return max_norm;
}

Eigen::VectorXd TrainReadout(const Eigen::MatrixXd &X_train, const Eigen::VectorXd &Y_train, double ridge_lambda)
{
    Eigen::MatrixXd XtX = X_train.transpose() * X_train;
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(XtX.rows(), XtX.cols());
    Eigen::VectorXd XtY = X_train.transpose() * Y_train;

    // Cambiamos ldlt() por colPivHouseholderQr() que es incondicionalmente estable frente a NaNs
    return (XtX + ridge_lambda * I).colPivHouseholderQr().solve(XtY);
}

double PearsonCorrelation(const vector<double> &y_true, const vector<double> &y_pred)
{
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
    int n = y_true.size();

    for (int i = 0; i < n; i++)
    {
        sum_x += y_true[i];
        sum_y += y_pred[i];
        sum_xy += y_true[i] * y_pred[i];
        sum_x2 += y_true[i] * y_true[i];
        sum_y2 += y_pred[i] * y_pred[i];
    }

    double var_x = n * sum_x2 - sum_x * sum_x;
    double var_y = n * sum_y2 - sum_y * sum_y;

    // BLINDAJE CONTRA EL NaN: Si la varianza es 0 o ligeramente negativa por redondeo, devuelve 0.
    if (var_x <= 0.0 || var_y <= 0.0)
    {
        return 0.0;
    }

    double numerator = n * sum_xy - sum_x * sum_y;
    double denominator = sqrt(var_x * var_y);

    //cout<<format("Pearson Correlation: numerator = {:.4f}, denominator = {:.4f}\n", numerator, denominator) << endl;

    return numerator / denominator;
}
double CalculateMemoryCapacity(const vector<vector<double>> &X_states, const vector<double> &predictions, int tau_vals, int size_input, double train_ratio)
{
    int washout = 100; // 1. Descartar transitorio inicial
    int train_size = (size_input * train_ratio) - washout;
    int test_size = size_input - (size_input * train_ratio);
    int res_size = X_states[0].size();

    // 2. Añadir +1 para la columna de BIAS
    Eigen::MatrixXd X_train(train_size, res_size + 1);
    Eigen::MatrixXd X_test(test_size, res_size + 1);

    // Rellenar X_train con Washout y Bias
    for (int i = 0; i < train_size; ++i)
    {
        for (int j = 0; j < res_size; ++j)
        {
            X_train(i, j) = X_states[washout + i][j];
        }
        X_train(i, res_size) = 1.0; // Nodo de Bias constante
    }

    int train_offset = washout + train_size;

    // Rellenar X_test con Bias
    for (int i = 0; i < test_size; ++i)
    {
        for (int j = 0; j < res_size; ++j)
        {
            X_test(i, j) = X_states[train_offset + i][j];
        }
        X_test(i, res_size) = 1.0; // Nodo de Bias constante
    }

    double memory_capacity = 0.0;

    for (int tau_idx = 0; tau_idx < tau_vals; ++tau_idx)
    {
        Eigen::VectorXd Y_train(train_size);
        vector<double> Y_test_vec(test_size);

        for (int t = 0; t < train_size; ++t)
        {
            Y_train(t) = predictions[tau_idx * size_input + washout + t];
        }
        for (int t = 0; t < test_size; ++t)
        {
            Y_test_vec[t] = predictions[tau_idx * size_input + train_offset + t];
        }

        Eigen::VectorXd W_out = TrainReadout(X_train, Y_train, 1e-4);
        Eigen::VectorXd Y_pred_eigen = X_test * W_out;

        vector<double> Y_pred_vec(test_size);
        for (int t = 0; t < test_size; ++t)
        {
            Y_pred_vec[t] = Y_pred_eigen(t);
        }

        double rho = PearsonCorrelation(Y_test_vec, Y_pred_vec);
        //cout<<format("Tau {}: Memory Capacity = {:.4f}\n", tau_idx + 1, rho * rho) << endl;

        // 3. Forzar flotante absoluto para que no capee a 0
        memory_capacity += std::abs(rho);
    }

    return memory_capacity;
}

double CalculateNARMAError(const vector<vector<double>> &X_states, const vector<double> &predictions, int size_input, double train_ratio)
{
    int washout = 100;
    int train_size = (size_input * train_ratio) - washout;
    int test_size = size_input - (size_input * train_ratio);
    int res_size = X_states[0].size();

    Eigen::MatrixXd X_train(train_size, res_size + 1);
    Eigen::MatrixXd X_test(test_size, res_size + 1);
    Eigen::VectorXd Y_train(train_size);
    vector<double> Y_test_vec(test_size);

    for (int i = 0; i < train_size; ++i)
    {
        for (int j = 0; j < res_size; ++j)
            X_train(i, j) = X_states[washout + i][j];
        X_train(i, res_size) = 1.0;
        Y_train(i) = predictions[washout + i];
    }

    int train_offset = washout + train_size;
    double mean_y = 0.0;

    for (int i = 0; i < test_size; ++i)
    {
        for (int j = 0; j < res_size; ++j)
            X_test(i, j) = X_states[train_offset + i][j];
        X_test(i, res_size) = 1.0;
        Y_test_vec[i] = predictions[train_offset + i];
        mean_y += Y_test_vec[i];
    }
    mean_y /= test_size;

    double var_y = 0.0;
    for (double y : Y_test_vec)
        var_y += (y - mean_y) * (y - mean_y);
    var_y /= test_size;

    Eigen::VectorXd W_out = TrainReadout(X_train, Y_train, 1e-4);
    Eigen::VectorXd Y_pred_eigen = X_test * W_out;

    double mse = 0.0;
    for (int i = 0; i < test_size; ++i)
    {
        double err = Y_test_vec[i] - Y_pred_eigen(i);
        mse += err * err;
    }
    mse /= test_size;

    if (var_y == 0.0)
        return 1.0;
    return sqrt(mse / var_y); // NRMSE
}